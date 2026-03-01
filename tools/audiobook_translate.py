#!/usr/bin/env python3
"""Audiobook translation pipeline using Chatterbox Multilingual TTS.

Translates an English audiobook into another language by:
1. Cloning the narrator's voice from the original audio
2. Synthesizing speech from translated ebook text
3. Assembling a complete audiobook with chapter structure
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
import torchaudio as ta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate an audiobook using Chatterbox Multilingual TTS"
    )
    parser.add_argument("--audio", required=True, help="Path to source audiobook (.m4b, .mp3, .wav)")
    parser.add_argument("--ebook", required=True, help="Path to translated ebook (.epub)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--lang", default="pt", help="Target language code (default: pt)")
    parser.add_argument("--ref-start", type=float, default=None, help="Voice reference start time in seconds")
    parser.add_argument("--ref-end", type=float, default=None, help="Voice reference end time in seconds")
    parser.add_argument("--resume", action="store_true", help="Resume from existing manifest")
    parser.add_argument("--chunk-silence", type=float, default=0.3, help="Silence between chunks in seconds (default: 0.3)")
    parser.add_argument("--paragraph-silence", type=float, default=1.0, help="Silence between paragraphs in seconds (default: 1.0)")
    parser.add_argument("--chapter-silence", type=float, default=2.0, help="Silence between chapters in seconds (default: 2.0)")
    parser.add_argument("--dry-run", action="store_true", help="Parse and chunk only, skip TTS generation")
    parser.add_argument("--max-chunks", type=int, default=None, help="Limit generation to N chunks (for testing)")
    parser.add_argument("--retry-failed", action="store_true", help="Reset failed chunks to pending for retry")
    parser.add_argument("--skip-chapters", type=str, default=None,
                        help="Comma-separated chapter numbers (1-indexed) to skip, e.g. '1,2,3,48,49'")
    parser.add_argument("--start-chapter", type=int, default=None,
                        help="First chapter number (1-indexed) to include")
    parser.add_argument("--end-chapter", type=int, default=None,
                        help="Last chapter number (1-indexed) to include")
    return parser.parse_args()


def extract_voice_reference(audio_path: str, output_dir: str, start: float | None, end: float | None) -> str:
    """Extract a voice reference clip from the source audiobook.

    If start/end are provided, extracts that segment.
    Otherwise, extracts a 10-second clip starting at 2 minutes
    (skipping intros/music).

    Returns path to the extracted reference.wav.
    """
    ref_path = os.path.join(output_dir, "reference.wav")

    if os.path.exists(ref_path):
        log.info(f"Voice reference already exists: {ref_path}")
        return ref_path

    if start is not None and end is not None:
        duration = end - start
        log.info(f"Extracting voice reference: {start}s to {end}s ({duration}s)")
    else:
        start = 120.0  # 2 minutes in — usually past intro music
        duration = 10.0
        log.info(f"No reference timestamps given, using default: {start}s to {start + duration}s")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", audio_path,
        "-ac", "1",         # mono
        "-ar", "24000",     # match Chatterbox output sample rate
        "-acodec", "pcm_s16le",
        ref_path,
    ]
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"ffmpeg failed:\n{result.stderr}")
        sys.exit(1)

    log.info(f"Voice reference saved: {ref_path}")
    return ref_path


def parse_epub(epub_path: str) -> list[dict]:
    """Parse an EPUB file and return chapters with their text.

    Returns a list of dicts: [{"title": str, "paragraphs": [str, ...]}]
    """
    from ebooklib import epub
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
    import warnings
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    book = epub.read_epub(epub_path, options={"ignore_ncx": True})
    chapters = []

    for item_id, _ in book.spine:
        item = book.get_item_with_id(item_id)
        if item is None:
            continue
        soup = BeautifulSoup(item.get_content(), "lxml")

        # Extract text from paragraph tags
        paragraphs = []
        for p in soup.find_all(["p", "h1", "h2", "h3"]):
            text = p.get_text(strip=True)
            if text and len(text) > 1:  # skip empty or single-char paragraphs
                paragraphs.append(text)

        if not paragraphs:
            continue

        # Try to extract chapter title from headings
        title_tag = soup.find(["h1", "h2", "h3"])
        title = title_tag.get_text(strip=True) if title_tag else f"Chapter {len(chapters) + 1}"

        chapters.append({"title": title, "paragraphs": paragraphs})

    log.info(f"Parsed {len(chapters)} chapters from EPUB")
    for i, ch in enumerate(chapters):
        log.info(f"  Chapter {i+1}: '{ch['title']}' — {len(ch['paragraphs'])} paragraphs")

    return chapters


def chunk_text(text: str, max_chars: int = 500) -> list[str]:
    """Split text into chunks suitable for TTS generation.

    Strategy:
    1. If text is under max_chars, return as-is
    2. Otherwise split at sentence boundaries
    3. If a sentence is still too long, split at clause boundaries
    """
    if len(text) <= max_chars:
        return [text]

    # Split at sentence boundaries
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_pattern.split(text)

    chunks = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            # Split long sentence at clause boundaries
            clause_pattern = re.compile(r'(?<=[,;])\s+|(?<=\s—\s)')
            clauses = clause_pattern.split(sentence)
            for clause in clauses:
                if current and len(current) + len(clause) + 1 > max_chars:
                    chunks.append(current.strip())
                    current = clause
                else:
                    current = f"{current} {clause}".strip() if current else clause
        elif current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks


def build_chunk_list(chapters: list[dict]) -> list[dict]:
    """Build a flat list of all text chunks with metadata.

    Returns: [{"chapter_idx": int, "chapter_title": str, "paragraph_idx": int,
               "chunk_idx": int, "text": str, "is_paragraph_end": bool}]
    """
    all_chunks = []

    for ch_idx, chapter in enumerate(chapters):
        for p_idx, paragraph in enumerate(chapter["paragraphs"]):
            chunks = chunk_text(paragraph)
            for c_idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "chapter_idx": ch_idx,
                    "chapter_title": chapter["title"],
                    "paragraph_idx": p_idx,
                    "chunk_idx": c_idx,
                    "text": chunk,
                    "is_paragraph_end": c_idx == len(chunks) - 1,
                })

    log.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks


def chunk_id(chunk: dict) -> str:
    """Generate a unique ID for a chunk."""
    return f"ch{chunk['chapter_idx']:03d}_p{chunk['paragraph_idx']:04d}_c{chunk['chunk_idx']:03d}"


def chunk_wav_path(output_dir: str, chunk: dict) -> str:
    """Return the .wav path for a given chunk."""
    ch_dir = os.path.join(output_dir, "chapters", f"ch{chunk['chapter_idx']:03d}")
    os.makedirs(ch_dir, exist_ok=True)
    return os.path.join(ch_dir, f"{chunk_id(chunk)}.wav")


def create_manifest(output_dir: str, chunks: list[dict]) -> dict:
    """Create a new manifest tracking all chunks."""
    manifest = {
        "version": 1,
        "chunks": {},
    }
    for chunk in chunks:
        cid = chunk_id(chunk)
        manifest["chunks"][cid] = {
            **chunk,
            "status": "pending",
            "wav_path": chunk_wav_path(output_dir, chunk),
        }
    return manifest


def load_manifest(output_dir: str) -> dict | None:
    """Load an existing manifest if it exists."""
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    return None


def save_manifest(output_dir: str, manifest: dict):
    """Save manifest to disk."""
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def generate_chunks(manifest: dict, output_dir: str, ref_path: str, lang: str, max_chunks: int | None = None):
    """Generate TTS audio for all pending chunks."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    log.info(f"Loading ChatterboxMultilingualTTS on {device}...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    log.info("Model loaded.")

    # Pre-compute voice conditionals once to avoid re-embedding every chunk
    log.info("Preparing voice conditionals from reference audio...")
    model.prepare_conditionals(ref_path)
    log.info("Voice conditionals ready.")

    pending = [(cid, c) for cid, c in manifest["chunks"].items() if c["status"] == "pending"]

    if max_chunks is not None:
        pending = pending[:max_chunks]
        log.info(f"  Limited to {max_chunks} chunks for testing")

    total = len(manifest["chunks"])
    done_before = total - len(pending)

    log.info(f"Generating {len(pending)} chunks ({done_before} already done)")

    for i, (cid, chunk) in enumerate(pending):
        progress = done_before + i + 1
        log.info(f"[{progress}/{total}] Generating {cid}: {chunk['text'][:60]}...")

        try:
            wav = model.generate(
                text=chunk["text"],
                language_id=lang,
                temperature=0.8,
                top_p=0.95,
            )

            wav_path = chunk["wav_path"]
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)
            ta.save(wav_path, wav, model.sr)

            manifest["chunks"][cid]["status"] = "done"
            log.info(f"  Saved: {wav_path}")

        except Exception as e:
            manifest["chunks"][cid]["status"] = "failed"
            log.error(f"  FAILED: {e}")

        # Save manifest after every chunk for resumability
        save_manifest(output_dir, manifest)

    done_total = sum(1 for c in manifest["chunks"].values() if c["status"] == "done")
    failed = sum(1 for c in manifest["chunks"].values() if c["status"] == "failed")
    log.info(f"Generation complete: {done_total} done, {failed} failed, {total} total")


def make_silence(duration_secs: float, sample_rate: int = 24000) -> torch.Tensor:
    """Create a silence tensor."""
    num_samples = int(duration_secs * sample_rate)
    return torch.zeros(1, num_samples)


def assemble_audiobook(manifest: dict, output_dir: str, chunk_silence: float, paragraph_silence: float, chapter_silence: float, lang: str = "pt"):
    """Concatenate all generated chunks into chapter files and a final audiobook."""
    sample_rate = 24000  # Chatterbox output SR

    # Group chunks by chapter
    chapters: dict[int, list] = {}
    for cid, chunk in sorted(manifest["chunks"].items()):
        if chunk["status"] != "done":
            continue
        ch_idx = chunk["chapter_idx"]
        if ch_idx not in chapters:
            chapters[ch_idx] = []
        chapters[ch_idx].append(chunk)

    if not chapters:
        log.error("No completed chunks to assemble")
        return

    for ch_idx in sorted(chapters.keys()):
        chunks = chapters[ch_idx]
        ch_title = chunks[0]["chapter_title"]
        log.info(f"Assembling chapter {ch_idx}: '{ch_title}' ({len(chunks)} chunks)")

        parts = []
        for chunk in chunks:
            wav, sr = ta.load(chunk["wav_path"])
            parts.append(wav)

            # Add appropriate silence
            if chunk["is_paragraph_end"]:
                parts.append(make_silence(paragraph_silence, sample_rate))
            else:
                parts.append(make_silence(chunk_silence, sample_rate))

        # Concatenate chapter in memory and save
        chapter_audio = torch.cat(parts, dim=1)
        chapter_path = os.path.join(output_dir, "chapters", f"ch{ch_idx:03d}.wav")
        ta.save(chapter_path, chapter_audio, sample_rate)
        log.info(f"  Saved chapter: {chapter_path} ({chapter_audio.shape[1] / sample_rate:.1f}s)")
        del chapter_audio, parts  # free memory before next chapter

    # Use ffmpeg concat to join chapter WAVs into final audiobook
    concat_list_path = os.path.join(output_dir, "concat_list.txt")
    silence_path = os.path.join(output_dir, "chapter_silence.wav")

    # Save chapter silence as a WAV file
    silence = make_silence(chapter_silence, sample_rate)
    ta.save(silence_path, silence, sample_rate)

    with open(concat_list_path, "w") as f:
        for i, ch_idx in enumerate(sorted(chapters.keys())):
            chapter_path = os.path.join(output_dir, "chapters", f"ch{ch_idx:03d}.wav")
            f.write(f"file '{os.path.abspath(chapter_path)}'\n")
            if i < len(chapters) - 1:  # no silence after last chapter
                f.write(f"file '{os.path.abspath(silence_path)}'\n")

    final_path = os.path.join(output_dir, f"audiobook_{lang}.wav")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path, "-c", "copy", final_path]
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"ffmpeg concat failed:\n{result.stderr}")
        return

    log.info(f"Final audiobook saved: {final_path}")


def main():
    args = parse_args()

    if (args.ref_start is None) != (args.ref_end is None):
        log.error("--ref-start and --ref-end must both be provided, or both omitted")
        sys.exit(1)

    log.info("Audiobook translation pipeline starting")
    log.info(f"  Audio: {args.audio}")
    log.info(f"  Ebook: {args.ebook}")
    log.info(f"  Output: {args.output}")
    log.info(f"  Language: {args.lang}")
    log.info(f"  Resume: {args.resume}")
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "chapters"), exist_ok=True)

    # Stage 1a: Extract voice reference
    ref_path = extract_voice_reference(args.audio, args.output, args.ref_start, args.ref_end)
    log.info(f"Using voice reference: {ref_path}")

    # Stage 1b: Parse EPUB
    chapters = parse_epub(args.ebook)
    if not chapters:
        log.error("No chapters found in EPUB")
        sys.exit(1)

    # Filter chapters if requested
    total_parsed = len(chapters)
    if args.skip_chapters:
        skip_set = {int(x.strip()) for x in args.skip_chapters.split(",")}
        chapters = [ch for i, ch in enumerate(chapters, 1) if i not in skip_set]
        log.info(f"Skipped chapters {skip_set}: {total_parsed} -> {len(chapters)} chapters")
    if args.start_chapter is not None or args.end_chapter is not None:
        start = (args.start_chapter or 1) - 1
        end = args.end_chapter or total_parsed
        chapters = chapters[start:end]
        log.info(f"Chapter range [{start+1}-{end}]: {len(chapters)} chapters selected")

    # Stage 1c: Build chunk list and manifest
    all_chunks = build_chunk_list(chapters)

    if args.resume:
        manifest = load_manifest(args.output)
        if manifest:
            done = sum(1 for c in manifest["chunks"].values() if c["status"] == "done")
            total = len(manifest["chunks"])
            log.info(f"Resuming: {done}/{total} chunks already done")
        else:
            log.warning("--resume given but no manifest found, starting fresh")
            manifest = create_manifest(args.output, all_chunks)
    else:
        manifest = create_manifest(args.output, all_chunks)

    if args.retry_failed:
        reset_count = 0
        for cid, chunk in manifest["chunks"].items():
            if chunk["status"] == "failed":
                chunk["status"] = "pending"
                reset_count += 1
        if reset_count:
            log.info(f"Reset {reset_count} failed chunks to pending for retry")

    save_manifest(args.output, manifest)
    log.info(f"Manifest saved with {len(manifest['chunks'])} chunks")

    if not args.dry_run:
        # Stage 2: Generate TTS audio
        generate_chunks(manifest, args.output, ref_path, args.lang, args.max_chunks)

        # Stage 3: Assemble audiobook
        assemble_audiobook(manifest, args.output, args.chunk_silence, args.paragraph_silence, args.chapter_silence, lang=args.lang)
    else:
        log.info("Dry run — skipping TTS generation and assembly")
        log.info(f"Would generate {len(manifest['chunks'])} chunks")

    log.info("Pipeline complete!")


if __name__ == "__main__":
    main()

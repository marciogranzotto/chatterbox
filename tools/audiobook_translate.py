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
import math
import os
import re
import subprocess
import sys
import time
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
    parser.add_argument("--repetition-penalty", type=float, default=2.5, help="Repetition penalty for TTS (default: 2.5, model default is 2.0)")
    parser.add_argument("--exaggeration", type=float, default=0.7, help="Expressiveness/emotion level (default: 0.7, model default is 0.5)")
    parser.add_argument("--cfg-weight", type=float, default=0.5, help="Classifier-free guidance weight (default: 0.5)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--skip-chapters", type=str, default=None,
                        help="Comma-separated chapter numbers (1-indexed) to skip, e.g. '1,2,3,48,49'")
    parser.add_argument("--start-chapter", type=int, default=None,
                        help="First chapter number (1-indexed) to include")
    parser.add_argument("--end-chapter", type=int, default=None,
                        help="Last chapter number (1-indexed) to include")
    parser.add_argument("--align-audio", action="store_true",
                        help="Use segment-aligned references: match each Portuguese chunk to its "
                             "corresponding position in the English audio for tone/emotion matching")
    parser.add_argument("--whisper-model", type=str, default="base",
                        help="Whisper model size for transcription (default: base). Options: tiny, base, small, medium, large")
    parser.add_argument("--ref-duration", type=float, default=10.0,
                        help="Duration in seconds for each aligned reference clip (default: 10.0)")
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


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"ffprobe failed: {result.stderr}")
        sys.exit(1)
    return float(result.stdout.strip())


def transcribe_audio(audio_path: str, output_dir: str, model_size: str = "base") -> list[dict]:
    """Transcribe the English audiobook with Whisper to get timestamped segments.

    Returns list of segments: [{"start": float, "end": float, "text": str}]
    Caches the transcript as JSON to avoid re-transcription.
    Transcribes in 30-minute chunks with checkpointing for resumability.
    """
    transcript_path = os.path.join(output_dir, "whisper_transcript.json")
    partial_path = os.path.join(output_dir, "whisper_transcript_partial.json")

    if os.path.exists(transcript_path):
        log.info(f"Loading cached Whisper transcript: {transcript_path}")
        with open(transcript_path, encoding="utf-8") as f:
            return json.load(f)

    # Convert to wav first for reliability (Whisper's ffmpeg handling can be fragile with m4b)
    wav_path = os.path.join(output_dir, "source_audio.wav")
    if not os.path.exists(wav_path):
        log.info("Converting source audio to WAV for Whisper...")
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-ac", "1",
            "-ar", "16000",
            "-acodec", "pcm_s16le",
            wav_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"ffmpeg conversion failed:\n{result.stderr}")
            sys.exit(1)
        log.info(f"Converted to: {wav_path}")

    # Get total duration and split into chunks
    total_duration = get_audio_duration(wav_path)
    CHUNK_MINUTES = 30
    chunk_duration = CHUNK_MINUTES * 60
    num_chunks = math.ceil(total_duration / chunk_duration)

    # Load partial progress if available
    segments = []
    completed_chunks = 0
    if os.path.exists(partial_path):
        with open(partial_path, encoding="utf-8") as f:
            partial = json.load(f)
        segments = partial["segments"]
        completed_chunks = partial["completed_chunks"]
        log.info(f"Resuming transcription from chunk {completed_chunks + 1}/{num_chunks} "
                 f"({completed_chunks * CHUNK_MINUTES}min already transcribed)")

    import whisper
    whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Transcribing with Whisper ({model_size}) on {whisper_device}...")
    log.info(f"  Total duration: {total_duration/3600:.1f}h in {num_chunks} chunks of {CHUNK_MINUTES}min")
    w_model = whisper.load_model(model_size, device=whisper_device)

    chunk_wav = os.path.join(output_dir, "whisper_chunk_temp.wav")

    for i in range(completed_chunks, num_chunks):
        start_time = i * chunk_duration
        remaining = min(chunk_duration, total_duration - start_time)
        log.info(f"Transcribing chunk {i + 1}/{num_chunks} "
                 f"({start_time/3600:.1f}h - {(start_time + remaining)/3600:.1f}h)...")

        # Extract chunk with ffmpeg
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-t", str(remaining),
            "-i", wav_path,
            "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
            chunk_wav,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"ffmpeg chunk extraction failed: {result.stderr}")
            sys.exit(1)

        # Transcribe chunk
        result = w_model.transcribe(chunk_wav, language="en", verbose=True,
                                    fp16=(whisper_device == "cuda"))

        # Add segments with time offset
        for seg in result["segments"]:
            segments.append({
                "start": seg["start"] + start_time,
                "end": seg["end"] + start_time,
                "text": seg["text"].strip(),
            })

        # Save checkpoint after each chunk
        completed_chunks = i + 1
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump({
                "completed_chunks": completed_chunks,
                "total_chunks": num_chunks,
                "chunk_duration_min": CHUNK_MINUTES,
                "segments": segments,
            }, f, indent=2)
        log.info(f"  Checkpoint saved: {completed_chunks}/{num_chunks} chunks done")

    del w_model

    # Clean up temp chunk
    if os.path.exists(chunk_wav):
        os.remove(chunk_wav)

    # Save final transcript
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    # Clean up partial checkpoint
    if os.path.exists(partial_path):
        os.remove(partial_path)

    total_transcribed = segments[-1]["end"] if segments else 0
    log.info(f"Transcription complete: {len(segments)} segments, {total_transcribed/3600:.1f} hours")

    # Clean up intermediate WAV (large file)
    if os.path.exists(wav_path):
        os.remove(wav_path)
        log.info("Cleaned up intermediate WAV file")

    return segments


def detect_chapter_boundaries(segments: list[dict], epub_chapters: list[dict]) -> list[dict]:
    """Detect chapter boundaries in the Whisper transcript by matching chapter titles.

    Returns list of chapter ranges: [{"title": str, "audio_start": float, "audio_end": float, "epub_idx": int}]
    """
    chapter_ranges = []

    # Build list of chapter titles to search for (normalized)
    titles = []
    for i, ch in enumerate(epub_chapters):
        # Normalize: lowercase, strip, collapse whitespace
        title = " ".join(ch["title"].lower().split())
        titles.append((i, title, ch["title"]))

    # Search for each title in the transcript segments
    # We look for segments whose text contains the chapter title
    total_duration = segments[-1]["end"] if segments else 0

    found_boundaries = []
    for epub_idx, norm_title, orig_title in titles:
        # Skip very short/generic titles
        if len(norm_title) < 3:
            continue

        for seg in segments:
            seg_text = " ".join(seg["text"].lower().split())
            # Check if segment text matches chapter title
            if norm_title in seg_text or seg_text in norm_title:
                found_boundaries.append({
                    "epub_idx": epub_idx,
                    "title": orig_title,
                    "audio_start": seg["start"],
                })
                break

    if not found_boundaries:
        # Fallback: divide audio equally among chapters
        log.warning("Could not detect chapter boundaries from transcript, using proportional split")
        chapter_duration = total_duration / len(epub_chapters)
        for i, ch in enumerate(epub_chapters):
            chapter_ranges.append({
                "title": ch["title"],
                "audio_start": i * chapter_duration,
                "audio_end": (i + 1) * chapter_duration,
                "epub_idx": i,
            })
        return chapter_ranges

    # Sort by audio position
    found_boundaries.sort(key=lambda x: x["audio_start"])

    # Fill in end times (each chapter ends where the next begins)
    for i, boundary in enumerate(found_boundaries):
        if i + 1 < len(found_boundaries):
            boundary["audio_end"] = found_boundaries[i + 1]["audio_start"]
        else:
            boundary["audio_end"] = total_duration
        chapter_ranges.append(boundary)

    log.info(f"Detected {len(chapter_ranges)} chapter boundaries in audio:")
    for cr in chapter_ranges[:5]:
        dur = (cr["audio_end"] - cr["audio_start"]) / 60
        log.info(f"  '{cr['title']}': {cr['audio_start']:.0f}s - {cr['audio_end']:.0f}s ({dur:.1f}min)")
    if len(chapter_ranges) > 5:
        log.info(f"  ... and {len(chapter_ranges) - 5} more")

    return chapter_ranges


def compute_aligned_references(manifest: dict, chapter_ranges: list[dict],
                                audio_path: str, output_dir: str,
                                ref_duration: float = 10.0) -> dict[str, str]:
    """Compute per-chunk reference audio clips based on proportional alignment.

    For each chunk, calculates its proportional position within its chapter,
    maps that to the corresponding position in the English audio, and extracts
    a reference clip.

    Returns: dict mapping chunk_id -> reference wav path
    """
    refs_dir = os.path.join(output_dir, "aligned_refs")
    os.makedirs(refs_dir, exist_ok=True)

    # Build a mapping from epub chapter index to audio range
    epub_to_audio = {}
    for cr in chapter_ranges:
        epub_to_audio[cr["epub_idx"]] = cr

    # Group chunks by chapter and count total text length per chapter
    chapter_chunks: dict[int, list[tuple[str, dict]]] = {}
    for cid, chunk in manifest["chunks"].items():
        ch_idx = chunk["chapter_idx"]
        if ch_idx not in chapter_chunks:
            chapter_chunks[ch_idx] = []
        chapter_chunks[ch_idx].append((cid, chunk))

    # Sort within each chapter
    for ch_idx in chapter_chunks:
        chapter_chunks[ch_idx].sort(key=lambda x: x[0])

    ref_map = {}
    total_audio_duration = max(cr["audio_end"] for cr in chapter_ranges)

    for ch_idx, chunks in chapter_chunks.items():
        # Find the audio range for this chapter
        if ch_idx in epub_to_audio:
            audio_range = epub_to_audio[ch_idx]
        else:
            # Chapter not found in audio — use proportional fallback
            n_chapters = max(chapter_chunks.keys()) + 1
            ch_start = (ch_idx / n_chapters) * total_audio_duration
            ch_end = ((ch_idx + 1) / n_chapters) * total_audio_duration
            audio_range = {"audio_start": ch_start, "audio_end": ch_end}

        ch_audio_start = audio_range["audio_start"]
        ch_audio_end = audio_range["audio_end"]
        ch_audio_duration = ch_audio_end - ch_audio_start

        # Calculate cumulative text length for proportional positioning
        total_text_len = sum(len(c["text"]) for _, c in chunks)
        cumulative_len = 0

        for cid, chunk in chunks:
            # Position of this chunk as fraction through the chapter
            fraction = cumulative_len / total_text_len if total_text_len > 0 else 0
            cumulative_len += len(chunk["text"])

            # Map to audio position
            audio_pos = ch_audio_start + fraction * ch_audio_duration

            # Center the reference clip around this position
            ref_start = max(0, audio_pos - ref_duration / 2)
            ref_end = ref_start + ref_duration
            # Don't exceed chapter bounds
            if ref_end > ch_audio_end:
                ref_end = ch_audio_end
                ref_start = max(0, ref_end - ref_duration)

            ref_path = os.path.join(refs_dir, f"{cid}_ref.wav")
            ref_map[cid] = {
                "path": ref_path,
                "start": ref_start,
                "end": ref_end,
            }

    # Extract all reference clips via ffmpeg (batch for efficiency)
    existing = sum(1 for v in ref_map.values() if os.path.exists(v["path"]))
    to_extract = {cid: v for cid, v in ref_map.items() if not os.path.exists(v["path"])}

    if existing:
        log.info(f"  {existing} aligned references already cached")

    if to_extract:
        log.info(f"Extracting {len(to_extract)} aligned reference clips...")
        for i, (cid, ref_info) in enumerate(to_extract.items()):
            if (i + 1) % 100 == 0 or i == 0:
                log.info(f"  Extracting reference {i+1}/{len(to_extract)}...")
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(ref_info["start"]),
                "-t", str(ref_info["end"] - ref_info["start"]),
                "-i", audio_path,
                "-ac", "1",
                "-ar", "24000",
                "-acodec", "pcm_s16le",
                ref_info["path"],
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                log.warning(f"  Failed to extract ref for {cid}: {result.stderr[:100]}")

    log.info(f"Aligned references ready: {len(ref_map)} clips")

    # Return just the path mapping
    return {cid: v["path"] for cid, v in ref_map.items()}


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
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_manifest(output_dir: str, manifest: dict, full_manifest: dict = None):
    """Save manifest to disk. If full_manifest is provided, merge filtered
    chunks back into the full manifest before saving (preserves chunks
    outside the current chapter selection)."""
    manifest_path = os.path.join(output_dir, "manifest.json")
    if full_manifest is not None:
        full_manifest["chunks"].update(manifest["chunks"])
        to_save = full_manifest
    else:
        to_save = manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=2, ensure_ascii=False)


def generate_chunks(manifest: dict, output_dir: str, ref_path: str, lang: str,
                    max_chunks: int | None = None, repetition_penalty: float = 2.5,
                    exaggeration: float = 0.7, cfg_weight: float = 0.5,
                    temperature: float = 0.8, aligned_refs: dict[str, str] | None = None,
                    full_manifest: dict = None):
    """Generate TTS audio for all pending chunks.

    If aligned_refs is provided, uses per-chunk reference clips instead of a single global reference.
    If full_manifest is provided, merges updates back when saving (for chapter-filtered resumes).
    """
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

    if aligned_refs:
        log.info("Using segment-aligned per-chunk references")
        # Prepare initial conditionals from global ref as fallback
        model.prepare_conditionals(ref_path, exaggeration=exaggeration)
    else:
        # Pre-compute voice conditionals once to avoid re-embedding every chunk
        log.info(f"Preparing voice conditionals from reference audio (exaggeration={exaggeration})...")
        model.prepare_conditionals(ref_path, exaggeration=exaggeration)
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
            # Use aligned reference if available, otherwise use global ref
            chunk_ref = None
            if aligned_refs and cid in aligned_refs:
                chunk_ref = aligned_refs[cid]
                if not os.path.exists(chunk_ref):
                    chunk_ref = None  # fall back to pre-computed conditionals

            if chunk_ref:
                wav = model.generate(
                    text=chunk["text"],
                    language_id=lang,
                    audio_prompt_path=chunk_ref,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    exaggeration=exaggeration,
                    repetition_penalty=repetition_penalty,
                )
            else:
                wav = model.generate(
                    text=chunk["text"],
                    language_id=lang,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    exaggeration=exaggeration,
                    repetition_penalty=repetition_penalty,
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
        save_manifest(output_dir, manifest, full_manifest=full_manifest)

    done_total = sum(1 for c in manifest["chunks"].values() if c["status"] == "done")
    failed = sum(1 for c in manifest["chunks"].values() if c["status"] == "failed")
    log.info(f"Generation complete: {done_total} done, {failed} failed, {total} total")


def make_silence(duration_secs: float, sample_rate: int = 24000) -> torch.Tensor:
    """Create a silence tensor."""
    num_samples = int(duration_secs * sample_rate)
    return torch.zeros(1, num_samples)


def trim_audio(wav: torch.Tensor, sample_rate: int = 24000, threshold: float = 0.01,
               silence_gap_ms: int = 1000) -> torch.Tensor:
    """Trim audio after the first long silence gap (generation artifacts).

    The TTS model sometimes generates valid speech, then a silence gap,
    then garbled re-generation. This detects the first silence gap longer
    than silence_gap_ms and cuts everything after it.

    Also trims trailing silence from the end.
    """
    window = int(0.05 * sample_rate)  # 50ms
    n_windows = wav.shape[1] // window
    gap_windows = silence_gap_ms // 50  # how many silent windows = a gap

    # Compute energy per window
    energies = []
    for i in range(n_windows):
        start = i * window
        end = (i + 1) * window
        rms = torch.sqrt(torch.mean(wav[0, start:end] ** 2)).item()
        energies.append(rms)

    # Find first silence gap longer than gap_windows consecutive silent windows
    # Only look after the first 500ms of audio (skip leading silence)
    silent_run = 0
    cut_point = None
    for i in range(10, n_windows):  # start at 500ms
        if energies[i] < threshold:
            silent_run += 1
            if silent_run >= gap_windows:
                # Found a long gap — cut at the start of this silence
                cut_point = (i - silent_run + 1) * window
                break
        else:
            silent_run = 0

    if cut_point is not None:
        trimmed = wav[:, :cut_point]
        trimmed_ms = (wav.shape[1] - trimmed.shape[1]) / sample_rate * 1000
        log.info(f"    Trimmed {trimmed_ms:.0f}ms of post-speech artifacts")
        return trimmed

    # No mid-audio gap found — just trim trailing silence
    last_active = n_windows
    for i in range(n_windows - 1, -1, -1):
        if energies[i] >= threshold:
            last_active = i + 1
            break

    tail_samples = int(0.2 * sample_rate)  # keep 200ms tail
    trim_point = min(last_active * window + tail_samples, wav.shape[1])
    trimmed = wav[:, :trim_point]

    trimmed_ms = (wav.shape[1] - trimmed.shape[1]) / sample_rate * 1000
    if trimmed_ms > 100:
        log.info(f"    Trimmed {trimmed_ms:.0f}ms trailing silence")

    return trimmed


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
            wav = trim_audio(wav, sample_rate)
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

    start_time = time.time()
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

    # Determine which chapter indices are selected
    selected_chapter_idxs = {c["chapter_idx"] for c in all_chunks}
    full_manifest = None  # Track full manifest when filtering on resume

    if args.resume:
        old_manifest = load_manifest(args.output)
        # Always build a fresh manifest from the current chunk list
        manifest = create_manifest(args.output, all_chunks)
        if old_manifest:
            # Carry over statuses from previous run for matching chunks
            carried = 0
            for cid in manifest["chunks"]:
                if cid in old_manifest["chunks"]:
                    old_status = old_manifest["chunks"][cid]["status"]
                    if old_status in ("done", "failed"):
                        manifest["chunks"][cid]["status"] = old_status
                        carried += 1

            # If old manifest has chunks outside current selection, preserve them for saving
            extra_cids = set(old_manifest["chunks"].keys()) - set(manifest["chunks"].keys())
            if extra_cids:
                full_manifest = old_manifest
                log.info(f"  {len(extra_cids)} chunks from other chapters preserved")

            done = sum(1 for c in manifest["chunks"].values() if c["status"] == "done")
            total = len(manifest["chunks"])
            log.info(f"Resuming: {done}/{total} chunks done ({carried} carried from previous run)")
        else:
            log.warning("--resume given but no manifest found, starting fresh")
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

    save_manifest(args.output, manifest, full_manifest=full_manifest)
    log.info(f"Manifest saved with {len(manifest['chunks'])} chunks")

    # Stage 1d: Segment alignment (optional)
    aligned_refs = None
    if args.align_audio:
        log.info("=== Segment-aligned mode: matching English audio to Portuguese chunks ===")
        transcript = transcribe_audio(args.audio, args.output, model_size=args.whisper_model)
        chapter_ranges = detect_chapter_boundaries(transcript, chapters)

        # Save chapter ranges for debugging
        ranges_path = os.path.join(args.output, "chapter_ranges.json")
        with open(ranges_path, "w", encoding="utf-8") as f:
            json.dump(chapter_ranges, f, indent=2, ensure_ascii=False)
        log.info(f"Chapter ranges saved: {ranges_path}")

        aligned_refs = compute_aligned_references(
            manifest, chapter_ranges, args.audio, args.output, ref_duration=args.ref_duration
        )

    if not args.dry_run:
        # Stage 2: Generate TTS audio
        generate_chunks(manifest, args.output, ref_path, args.lang, args.max_chunks,
                        args.repetition_penalty, args.exaggeration, args.cfg_weight,
                        args.temperature, aligned_refs=aligned_refs,
                        full_manifest=full_manifest)

        # Stage 3: Assemble audiobook
        assemble_audiobook(manifest, args.output, args.chunk_silence, args.paragraph_silence, args.chapter_silence, lang=args.lang)
    else:
        log.info("Dry run — skipping TTS generation and assembly")
        log.info(f"Would generate {len(manifest['chunks'])} chunks")
        if aligned_refs:
            log.info(f"Would use {len(aligned_refs)} aligned reference clips")

    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    log.info(f"Pipeline complete! Total time: {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    main()

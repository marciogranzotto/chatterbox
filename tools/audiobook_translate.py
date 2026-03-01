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


def main():
    args = parse_args()
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


if __name__ == "__main__":
    main()

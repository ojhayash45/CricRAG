#!/usr/bin/env python
"""
Chunk inspection tool -- for debugging chunk quality.

    python scripts/inspect_chunks.py
    python scripts/inspect_chunks.py --count 20
    python scripts/inspect_chunks.py --law 36

Reads data/chunks/chunks.json (produced by scripts/ingest.py) and prints a
sample of chunks so you can eyeball whether section-aware chunking is
producing sensible, self-contained pieces of text.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.schemas import Chunk  # noqa: E402
from src.pipeline.ingest_runner import load_saved_chunks  # noqa: E402


def load_chunks() -> list[Chunk]:
    try:
        return load_saved_chunks()
    except RuntimeError as exc:
        print(f"{exc} Run scripts/ingest.py first.")
        return []


def print_chunk(chunk: Chunk, text_limit: int) -> None:
    text = chunk.text if len(chunk.text) <= text_limit else chunk.text[:text_limit] + "..."
    print("-" * 70)
    print(f"Chunk ID:  {chunk.chunk_id}")
    print(f"Law:       {chunk.law_number} - {chunk.law_title}")
    print(f"Section:   {chunk.section}")
    print(f"Edition:   {chunk.edition}")
    print(f"Source:    {chunk.source_url}")
    print(f"Text:\n{text}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a random sample of indexed chunks.")
    parser.add_argument("--count", type=int, default=20, help="Number of chunks to sample (default: 20)")
    parser.add_argument("--law", type=str, default=None, help="Only sample chunks from this law number")
    parser.add_argument("--text-limit", type=int, default=400, help="Max characters of chunk text to print")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, for reproducible sampling")
    args = parser.parse_args()

    chunks = load_chunks()
    if not chunks:
        return 1

    if args.law:
        chunks = [c for c in chunks if c.law_number == args.law]
        if not chunks:
            print(f"No chunks found for law {args.law!r}.")
            return 1

    if args.seed is not None:
        random.seed(args.seed)

    sample_size = min(args.count, len(chunks))
    sample = random.sample(chunks, sample_size)

    for chunk in sample:
        print_chunk(chunk, args.text_limit)

    print("-" * 70)
    print(f"Showed {sample_size} of {len(chunks)} total chunk(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

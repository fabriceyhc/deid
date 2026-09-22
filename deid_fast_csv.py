"""
Fast vectorized regex de-identification for large CSVs.
Applies all RegexMasker patterns using pandas vectorized str.replace — no per-cell Python loops.
"""

import re
import pandas as pd
import argparse
from tqdm import tqdm

PATTERNS = [
    (re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b"), "[REDACTED]"),
    (re.compile(r"(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])[-/]((?:19|20)\d{2}|\d{2})"), "[REDACTED]"),
    (re.compile(r"\d{3}-\d{2}-\d{4}"), "[REDACTED]"),
    (re.compile(r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?=[^0-9]|$)"), "[REDACTED]"),
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}(?=[^a-zA-Z0-9._%+\-]|$)"), "[REDACTED]"),
    (re.compile(r"(?:MRN\s*\d{5,}|\d{7,})"), "[REDACTED]"),
    (re.compile(r"\d{1,5}\s+(?:[A-Za-z0-9]+\s?){2,8}(?:,\s?)?[A-Za-z]+(?:[ -][A-Za-z]+)*(?:,\s?)?[A-Z]{2}(?:,?\s?\d{5})?"), "[REDACTED]"),
    (re.compile(r"\b(?:DWR|DWA|MD|Dr\.?)\s+([A-Z][a-z]+)\s+(?:DWR|DWA|MD|Dr\.?|[A-Z]{2,4})\b"), "[REDACTED]"),
    # Collapse consecutive [REDACTED] tags
    (re.compile(r"(\[REDACTED\])(?:\s*\[REDACTED\])+"), r"\1"),
]


def apply_patterns(series: pd.Series, n_passes: int = 2) -> pd.Series:
    s = series.fillna("").astype(str)
    for _ in range(n_passes):
        for pattern, replacement in PATTERNS:
            s = s.str.replace(pattern, replacement, regex=True)
    return s


def main():
    parser = argparse.ArgumentParser(description="Fast vectorized regex de-identification for large CSVs.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--column_name", required=True)
    parser.add_argument("--chunk_size", type=int, default=100_000)
    parser.add_argument("--num_passes", type=int, default=2)
    args = parser.parse_args()

    chunk_iter = pd.read_csv(args.input_csv, chunksize=args.chunk_size, low_memory=False)
    first = True

    for chunk in tqdm(chunk_iter, desc="Chunks"):
        if args.column_name not in chunk.columns:
            raise ValueError(f"Column '{args.column_name}' not found. Available: {chunk.columns.tolist()}")
        chunk[args.column_name] = apply_patterns(chunk[args.column_name], n_passes=args.num_passes)
        chunk.to_csv(args.output_csv, mode="w" if first else "a", header=first, index=False)
        first = False

    print(f"Done. Output saved to {args.output_csv}")


if __name__ == "__main__":
    main()

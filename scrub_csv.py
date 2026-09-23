#!/usr/bin/env python3
"""De-identify named text columns of an arbitrary CSV, preserving everything else.

run_deid_notes.py handles the line-per-note corpus layout. This is the general
case: annotation files, review sheets, evaluation sets -- any table with one or
more free-text columns alongside labels that must survive untouched.

If the table carries a patient id column, the self-referential layer is used for
those rows, which is the highest-precision signal available.

    python3 scrub_csv.py --input-csv IN.csv --output-csv OUT.csv \
        --text-columns NOTE_TEXT,rationale \
        --identifiers-csv Patient_Identifiers.csv [--gpu 7 | --no-ner]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--text-columns", default=None,
                    help="Comma-separated columns to scrub. Default: auto-detect "
                         "object columns whose mean length exceeds --min-len")
    ap.add_argument("--min-len", type=float, default=40,
                    help="Mean length above which a column counts as free text")
    ap.add_argument("--patient-id-column", default="IP_PATIENT_ID")
    ap.add_argument("--identifiers-csv", default=None)
    ap.add_argument("--roster", default="deid2/resources/provider_roster.txt")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--no-ner", action="store_true")
    ap.add_argument("--force-gpus", action="store_true")
    ap.add_argument("--cache-dir", default="/local1/fabricehc/huggingface/hub")
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--repair-only", action="store_true",
                    help="Apply only the post-mask given-name rule. For text that "
                         "has already been scrubbed, running the full rule set "
                         "again costs ~25 regexes per row for nothing; this mode "
                         "applies the one rule that closes the known residual.")
    ap.add_argument("--in-place", action="store_true",
                    help="Replace the input file once the output is written")
    args = ap.parse_args()

    device = "cpu"
    # --repair-only applies one regex rule and needs no model, so it must not
    # demand a GPU.
    if not args.no_ner and not args.repair_only:
        from deid2.gpu import check_gpus_free, select_visible
        if args.gpu is None:
            raise SystemExit("--gpu is required unless --no-ner is passed")
        if not args.force_gpus:
            check_gpus_free([args.gpu])
        device = select_visible(args.gpu)

    from deid2 import pipeline
    if args.repair_only:
        from deid2.pipeline import Deidentifier2, load_allowlists, load_given_names
        from deid2.rules import ClinicalRuleMasker
        clinical, general = load_allowlists()
        masker = ClinicalRuleMasker(given_names=load_given_names(),
                                    clinical_allowlist=clinical,
                                    general_allowlist=general)
        # Only the post-mask given-name rule; the rest of the table is skipped.
        masker.get_spans = masker._given_name_after_mask
        deid = Deidentifier2(patient_gazetteer=None, roster=None,
                             rules=masker, ner=None)
    else:
        deid = pipeline.build(
            identifiers_csv=args.identifiers_csv,
            roster_path=args.roster,
            use_ner=not args.no_ner,
            cache_dir=args.cache_dir,
            device=device,
            batch_size=args.batch_size,
        )

    first = True
    n_rows = 0
    cols_used = None
    out_path = pathlib.Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for chunk in pd.read_csv(args.input_csv, chunksize=args.chunk_size, low_memory=False):
        if cols_used is None:
            if args.text_columns:
                cols_used = [c for c in args.text_columns.split(",") if c in chunk.columns]
            else:
                cols_used = [c for c in chunk.columns
                             if chunk[c].dtype == object
                             and chunk[c].fillna("").astype(str).str.len().mean() > args.min_len]
            print(f"scrubbing columns: {cols_used}", file=sys.stderr)
            if not cols_used:
                raise SystemExit("no text columns found; pass --text-columns")

        pids = (chunk[args.patient_id_column].tolist()
                if args.patient_id_column in chunk.columns else [None] * len(chunk))
        for col in cols_used:
            texts = chunk[col].fillna("").astype(str).tolist()
            chunk[col] = deid.deidentify_batch(texts, pids)
        chunk.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
        first = False
        n_rows += len(chunk)
        print(f"\r  {n_rows:,} rows", end="", file=sys.stderr)
    print(file=sys.stderr)

    if deid.stats:
        print(deid.stats.render())
    if args.in_place:
        out_path.replace(args.input_csv)
        print(f"replaced {args.input_csv} in place", file=sys.stderr)


if __name__ == "__main__":
    main()

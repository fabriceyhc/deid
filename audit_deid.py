#!/usr/bin/env python3
"""Measure residual PHI in a notes corpus, before or after de-identification.

The headline metric needs no manual annotation. Because the study holds a
patient -> name crosswalk, the share of notes still containing their own
patient's name is a direct recall measurement against known ground truth. A
matched false-positive measurement is reported alongside it: the share of text
redacted, and the most frequently redacted surface forms, which is where
over-redaction shows up.

    python3 audit_deid.py --notes-csv NOTES.csv \
        --identifiers-csv Patient_Identifiers.csv [--sample-rows 200000]
"""

from __future__ import annotations

import argparse
import collections
import re
import sys

import pandas as pd

WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")

# Residual-PHI probes. These are deliberately broad: this is a leak detector,
# not the redactor, so recall matters more than precision here.
PROBES = {
    "DATE_numeric":  re.compile(r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)?\d{2}\b"),
    "DATE_monthname": re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2}\b", re.I),
    "PHONE":         re.compile(r"(?<![\d-])(?:\+?1[-.\s]?)?(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]?\d{4}(?![\d-])"),
    "SSN":           re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "EMAIL":         re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "URL":           re.compile(r"\bhttps?://\S+|\bwww\.[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "MRN_labeled":   re.compile(r"\b(?:MRN|MR#|Medical Record(?: Number)?)\s*[:#]?\s*\d{4,}\b", re.I),
    "AGE_over89":    re.compile(r"\b(?:9\d|1\d\d)\s*(?:y/?o|yo|year[- ]old|yrs? old)\b", re.I),
    "ZIP_in_state":  re.compile(r"\b(?:CA|California)[ ,]+9\d{4}\b"),
    "STREET_ADDR":   re.compile(r"\b\d{1,5}\s+(?:[NSEW]\.?\s+)?[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+(?:St|Street|Ave|Avenue|Blvd|Boulevard|Rd|Road|Dr|Drive|Ln|Lane|Ct|Court|Way|Pl|Place|Pkwy|Hwy|Cir|Circle)\b\.?"),
    "SALUTATION":    re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Miss)\.?\s+[A-Z][a-z]{2,}"),
    "NAME_CREDENTIAL": re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]{2,},?\s*(?:M\.?D\.?|D\.?O\.?|N\.?P\.?|R\.?N\.?)\b"),
    "PAGER_EXT":     re.compile(r"\b(?:pager|pgr|beeper|ext(?:ension)?)\s*[:#]?\s*\d{3,6}\b", re.I),
}

# Only this pass's own tags -- clinical notes contain bracketed words of their
# own ("[DISCONTINUED]", "[CT]") that are not redactions.
TAG = re.compile(r"\[(?:NAME|DATE|AGE|PHONE|EMAIL|ID|ADDRESS|CITY|ZIP|LOC|HOSP|URL|REDACTED)\]")
UPSTREAM_MASK = re.compile(r"_{3,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes-csv", required=True)
    ap.add_argument("--identifiers-csv", required=True)
    ap.add_argument("--text-column", default="NOTE_TEXT")
    ap.add_argument("--sample-rows", type=int, default=200_000)
    ap.add_argument("--note-ids-from", default=None,
                    help="Restrict the audit to the IP_NOTE_IDs present in this CSV, "
                         "so a before/after comparison covers the same notes")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ids = pd.read_csv(args.identifiers_csv)
    pat_tokens = {}
    for pid, nm in zip(ids["IP_PATIENT_ID"], ids["PAT_NAME"].fillna("")):
        toks = [t for t in re.split(r"[,\s]+", str(nm)) if len(t) >= 4 and t.isalpha()]
        if toks:
            pat_tokens[pid] = toks

    keep_ids = None
    if args.note_ids_from:
        keep_ids = set(pd.read_csv(args.note_ids_from,
                                   usecols=["IP_NOTE_ID"])["IP_NOTE_ID"].unique())
        print(f"restricting to {len(keep_ids):,} note ids", file=sys.stderr)

    counts = collections.Counter()
    notes_with = collections.Counter()
    redacted_surface = collections.Counter()
    n = n_pt_any = n_pt_two = 0
    leaked_patients = set()
    seen_patients = set()
    total_chars = tag_chars = 0

    cols = ["IP_PATIENT_ID", args.text_column]
    if keep_ids is not None:
        cols.append("IP_NOTE_ID")
    for chunk in pd.read_csv(args.notes_csv, chunksize=20_000,
                             usecols=cols, low_memory=False):
        if keep_ids is not None:
            chunk = chunk[chunk["IP_NOTE_ID"].isin(keep_ids)]
            if chunk.empty:
                continue
        for pid, txt in zip(chunk["IP_PATIENT_ID"], chunk[args.text_column].fillna("")):
            n += 1
            txt = str(txt)
            seen_patients.add(pid)
            total_chars += len(txt)
            for m in TAG.finditer(txt):
                tag_chars += len(m.group())
                redacted_surface[m.group()] += 1

            hits = set()
            for cat, rx in PROBES.items():
                found = rx.findall(txt)
                if found:
                    counts[cat] += len(found)
                    hits.add(cat)

            toks = pat_tokens.get(pid)
            if toks:
                present = [t for t in toks
                           if re.search(r"\b" + re.escape(t) + r"\b", txt, re.I)]
                if present:
                    n_pt_any += 1
                    hits.add("PATIENT_OWN_NAME")
                    leaked_patients.add(pid)
                if len(present) >= 2:
                    n_pt_two += 1
            for c in hits:
                notes_with[c] += 1
        if n >= args.sample_rows:
            break

    label = args.label or args.notes_csv
    out = [f"RESIDUAL PHI AUDIT: {label}",
           "=" * 72,
           f"rows scanned      : {n:,}",
           f"distinct patients : {len(seen_patients):,}",
           f"chars scanned     : {total_chars:,}",
           f"chars in [TAG]s   : {tag_chars:,} ({100*tag_chars/max(total_chars,1):.3f}% of text)",
           "",
           "PATIENT'S OWN NAME (ground truth from Patient_Identifiers.csv)",
           "-" * 72,
           f"  notes with >=1 of the patient's name tokens : {n_pt_any:,} ({100*n_pt_any/max(n,1):.3f}%)",
           f"  notes with >=2 of the patient's name tokens : {n_pt_two:,} ({100*n_pt_two/max(n,1):.3f}%)",
           f"  distinct patients with a leak              : {len(leaked_patients):,}"
           f" of {len(seen_patients):,} ({100*len(leaked_patients)/max(len(seen_patients),1):.2f}%)",
           "",
           "PATTERN PROBES",
           "-" * 72,
           f"  {'category':<20s} {'hits':>10s} {'notes':>10s} {'% notes':>9s}"]
    for cat in PROBES:
        out.append(f"  {cat:<20s} {counts[cat]:>10,} {notes_with[cat]:>10,}"
                   f" {100*notes_with[cat]/max(n,1):>8.3f}%")

    if redacted_surface:
        out += ["", "REDACTION TAGS APPLIED", "-" * 72]
        for tag, c in redacted_surface.most_common():
            out.append(f"  {tag:<12s} {c:>12,}")

    report = "\n".join(out)
    print(report)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(report + "\n")


if __name__ == "__main__":
    main()

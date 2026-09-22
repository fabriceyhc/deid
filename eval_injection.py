#!/usr/bin/env python3
"""Large-scale validation by PHI injection.

The patient-name crosswalk in audit_deid.py measures recall for one category
against real ground truth, but it says nothing about the other categories and
nothing about precision. This harness closes both gaps at scale:

  RECALL  -- known PHI is injected into real notes at known character offsets,
             using templates drawn from the leak contexts actually observed in
             this corpus, then scored span-exactly by category. Tens of
             thousands of labelled cases, no annotation.

  PRECISION -- measured separately as clinical-term preservation. Injected text
             cannot measure precision honestly, because the substrate notes
             still contain residual real PHI and a redaction there is correct,
             not a false positive. Instead, a curated list of terms that must
             never be redacted (the study's exposure vocabulary, common drugs,
             syndromes, vitals) is counted before and after. Any loss is a
             false positive.

    python3 eval_injection.py --notes-csv NOTES.csv --identifiers-csv IDS.csv \
        --n-notes 4000 --injections-per-note 10 [--no-ner] [--gpu 7]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import re
import sys
from typing import Dict, List, Sequence, Tuple

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# Terms that must survive de-identification. Losing any of these would quietly
# damage the downstream substance-use analysis.
MUST_SURVIVE = """
fentanyl norfentanyl methamphetamine amphetamine methadone buprenorphine
suboxone naloxone narcan naltrexone heroin cocaine benzodiazepine alcohol
cannabis marijuana opioid opiate injection intravenous ivdu
sepsis bacteremia endocarditis osteomyelitis cellulitis abscess myositis
meningitis discitis arthritis pneumonia empyema
vancomycin ceftriaxone cefepime zosyn piperacillin daptomycin linezolid
clindamycin metronidazole azithromycin doxycycline gentamicin
mrsa mssa staphylococcus streptococcus enterococcus pseudomonas candida
hiv hcv hbv aids cirrhosis
""".split()


# Whole-word matching: "archive" contains "hiv" and "candidate" contains
# "candida", so substring counting would report phantom losses.
TERM_RX = {t: re.compile(r"\b" + re.escape(t) + r"\b") for t in MUST_SURVIVE}


def load_name_pools(names_json: str) -> Tuple[List[str], List[str]]:
    data = json.load(open(names_json))
    first = list(data.get("men", [])) + list(data.get("women", []))
    last = list(data.get("last", []))
    return first, last


class PHIGenerator:
    """Synthesises PHI values and the templates they appear in."""

    STREETS = ["Maple", "Oak", "Cedar", "Pine", "Elm", "Sunset", "Vermont",
               "Figueroa", "Sepulveda", "Wilshire", "Olympic", "Pico",
               "Westwood", "Midvale", "Gayley", "Barrington", "Cloverdale"]
    SUFFIX = ["St", "Street", "Ave", "Avenue", "Blvd", "Rd", "Dr", "Ln", "Way", "Pl"]
    CITIES = ["Los Angeles", "Santa Monica", "Pasadena", "Torrance", "Inglewood",
              "Glendale", "Burbank", "Downey", "Norwalk", "Culver City"]
    MONTHS = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]

    def __init__(self, first: Sequence[str], last: Sequence[str], rng: random.Random):
        self.first, self.last, self.rng = list(first), list(last), rng

    def name(self) -> str:
        r = self.rng
        f, l = r.choice(self.first), r.choice(self.last)
        style = r.random()
        if style < 0.55:
            return f"{f} {l}"
        if style < 0.75:
            return f"{f} {r.choice('ABCDEFGHJKLMNPRSTW')}. {l}"
        if style < 0.85:
            return f"{l}, {f}"
        if style < 0.93:
            return f"{f} {l}-{r.choice(self.last)}"
        prefix = r.choice(["Mc", "Mac", "Van ", "O'"])
        return f"{f} {prefix}{r.choice(self.last)}"

    def surname(self) -> str:
        return self.rng.choice(self.last)

    def date(self) -> str:
        r = self.rng
        m, d, y = r.randint(1, 12), r.randint(1, 28), r.randint(1940, 2025)
        style = r.random()
        if style < 0.45:
            return f"{m}/{d}/{y}"
        if style < 0.60:
            return f"{m:02d}/{d:02d}/{y}"
        if style < 0.75:
            return f"{m}-{d}-{y}"
        if style < 0.90:
            return f"{self.MONTHS[m-1]} {d}, {y}"
        return f"{self.MONTHS[m-1][:3]} {d}, {y}"

    def phone(self) -> str:
        r = self.rng
        a, b, c = r.randint(200, 989), r.randint(200, 999), r.randint(1000, 9999)
        return r.choice([f"{a}-{b}-{c}", f"({a}) {b}-{c}", f"{a}.{b}.{c}",
                         f"+1 {a} {b} {c}", f"1-{a}-{b}-{c}"])

    def pager(self) -> str:
        return str(self.rng.randint(10000, 99999))

    def mrn(self) -> str:
        return str(self.rng.randint(1000000, 99999999))

    def email(self) -> str:
        r = self.rng
        return (f"{r.choice(self.first).lower()}.{r.choice(self.last).lower()}"
                f"@{r.choice(['gmail.com','yahoo.com','ucla.edu','mednet.ucla.edu'])}")

    def address(self) -> str:
        r = self.rng
        base = f"{r.randint(100, 9999)} {r.choice(self.STREETS)} {r.choice(self.SUFFIX)}"
        if r.random() < 0.4:
            base += f" {r.choice(['Apt','#','Unit','Ste'])} {r.randint(1, 450)}"
        return base

    def city_zip(self) -> str:
        return f"{self.rng.choice(self.CITIES)} CA {self.rng.randint(90001, 93599)}"

    def old_age(self) -> str:
        r = self.rng
        return f"{r.randint(90, 104)} {r.choice(['y.o.', 'yo', 'year-old', 'years old'])}"


# Each template is a list of literal strings and (category, generator) pairs.
def build_templates(g: PHIGenerator):
    N, D, P, E, I, A, Z, G = "NAME", "DATE", "PHONE", "EMAIL", "ID", "ADDRESS", "ZIP", "AGE"
    return [
        ["PATIENT: ", (N, g.name), "  MRN: ", (I, g.mrn), "  DOB: ", (D, g.date)],
        ["Pt. Name/Age/DOB:  ", (N, g.name), "   ", (G, g.old_age), "   ", (D, g.date)],
        ["Name: ", (N, g.name), "  MRN: ", (I, g.mrn), "  Date of Service: ", (D, g.date)],
        ["Attending: ", (N, g.name), ", MD   Resident: ", (N, g.name), ", MD"],
        ["Physician ordering/referring: ", (N, g.name), ", MSN"],
        ["Electronically signed by ", (N, g.name), ", MD on ", (D, g.date)],
        ["Dr. ", (N, g.surname), " was consulted and agreed with the plan."],
        ["Briefly, Mr. ", (N, g.name), " is a ", (G, g.old_age), " male admitted for sepsis."],
        ["Discharge Address: ", (A, g.address), ", ", (Z, g.city_zip)],
        ["Home address is ", (A, g.address), " in ", (Z, g.city_zip), "."],
        ["Contact Name: ", (N, g.name), "  Phone: ", (P, g.phone)],
        ["Emergency Contact: ", (N, g.name), " (", (P, g.phone), ")"],
        ["Next of Kin: ", (N, g.name), ", reachable at ", (P, g.phone)],
        ["Email: ", (E, g.email), "  Patient permission to contact confirmed."],
        ["Author: ", (N, g.name), ", RD, pager ", (P, g.pager)],
        ["The patient's daughter ", (N, g.name), " was present at bedside."],
        ["His wife ", (N, g.surname), " reports he has been more confused since ", (D, g.date), "."],
        ["Primary care physician: ", (N, g.name), ", MD  Clinic phone ", (P, g.phone)],
        ["Follow up with ", (N, g.name), ", NP on ", (D, g.date), " at the clinic."],
        ["PCP ", (N, g.name), " notified of admission on ", (D, g.date), "."],
    ]


def render(template) -> Tuple[str, List[Tuple[int, int, str]]]:
    """Return the filled template plus the gold spans inside it."""
    out, gold = [], []
    pos = 0
    for part in template:
        if isinstance(part, str):
            out.append(part)
            pos += len(part)
        else:
            cat, fn = part
            val = fn()
            gold.append((pos, pos + len(val), cat))
            out.append(val)
            pos += len(val)
    return "".join(out), gold


# Field-like seams in these single-line notes, where an injection looks native.
_SEAM = re.compile(r"\s{2,}")


def inject(note: str, templates, rng: random.Random, k: int):
    """Insert k rendered templates into a note; return new text and gold spans."""
    seams = [m.end() for m in _SEAM.finditer(note)]
    if len(seams) < k:
        seams += [rng.randrange(len(note) + 1) for _ in range(k - len(seams))]
    points = sorted(rng.sample(seams, min(k, len(seams))), reverse=True)

    text = note
    gold: List[Tuple[int, int, str]] = []
    placed: List[Tuple[int, str, List[Tuple[int, int, str]]]] = []
    for p in points:
        snippet, sgold = render(rng.choice(templates))
        snippet = snippet + "   "
        placed.append((p, snippet, sgold))

    # Insert from the end so earlier offsets stay valid, then shift the gold
    # spans of everything already placed to the right of each insertion.
    for p, snippet, sgold in placed:
        text = text[:p] + snippet + text[p:]
        gold = [(s + len(snippet), e + len(snippet), c) if s >= p else (s, e, c)
                for s, e, c in gold]
        gold.extend((p + s, p + e, c) for s, e, c in sgold)
    return text, gold


_ALNUM = re.compile(r"[A-Za-z0-9]")


def score(gold, predicted, text: str) -> Tuple[str, str]:
    """Classify one gold span as full / partial / missed.

    Coverage is assessed over the *union* of predicted spans, and only over the
    alphanumeric characters of the gold span. A name written "Vancamp, Arvin"
    is legitimately redacted as two spans with the comma left in place; judging
    it against a single containing span would score that as a partial miss.
    """
    start, end = gold[0], gold[1]
    label = ""
    covered = bytearray(end - start)
    for p in predicted:
        if p.start < end and p.end > start:
            label = label or p.label
            for i in range(max(p.start, start), min(p.end, end)):
                covered[i - start] = 1
    need = [i for i in range(start, end) if _ALNUM.match(text[i])]
    if not need:
        return "full", label
    hit = sum(1 for i in need if covered[i - start])
    if hit == len(need):
        return "full", label
    if hit:
        return "partial", label
    return "missed", ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notes-csv", required=True)
    ap.add_argument("--identifiers-csv", default=None)
    ap.add_argument("--roster", default="deid2/resources/provider_roster.txt")
    ap.add_argument("--names-json", default="data/names.json")
    ap.add_argument("--n-notes", type=int, default=4000)
    ap.add_argument("--injections-per-note", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-ner", action="store_true")
    ap.add_argument("--ner-models", default=None,
                    help="Comma-separated NER checkpoints, or 'none' to use GLiNER only")
    ap.add_argument("--gliner-model", default=None)
    ap.add_argument("--gliner-threshold", type=float, default=0.5)
    ap.add_argument("--gpu", type=int, default=None,
                    help="GPU index to pin to; omit for CPU")
    ap.add_argument("--force-gpus", action="store_true")
    ap.add_argument("--cache-dir", default="/local1/fabricehc/huggingface/hub")
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Pin the device before anything imports CUDA, and never land on a GPU
    # another user is working on.
    device = "cpu"
    if not args.no_ner:
        from deid2.gpu import check_gpus_free, select_visible
        if args.gpu is None:
            raise SystemExit("--gpu is required unless --no-ner is passed")
        if not args.force_gpus:
            check_gpus_free([args.gpu])
        device = select_visible(args.gpu)

    rng = random.Random(args.seed)
    first, last = load_name_pools(args.names_json)
    gen = PHIGenerator(first, last, rng)
    templates = build_templates(gen)

    print(f"loading up to {args.n_notes:,} notes...", file=sys.stderr)
    notes, pids = [], []
    for chunk in pd.read_csv(args.notes_csv, chunksize=20_000,
                             usecols=["IP_PATIENT_ID", "NOTE_TEXT"], low_memory=False):
        for pid, txt in zip(chunk["IP_PATIENT_ID"], chunk["NOTE_TEXT"].fillna("")):
            t = str(txt)
            if len(t) < 400:
                continue
            notes.append(t)
            pids.append(pid)
            if len(notes) >= args.n_notes:
                break
        if len(notes) >= args.n_notes:
            break

    from deid2 import pipeline
    deid = pipeline.build(
        identifiers_csv=args.identifiers_csv,
        roster_path=args.roster,
        use_ner=not args.no_ner,
        ner_models=args.ner_models.split(",") if args.ner_models else None,
        gliner_model=args.gliner_model,
        gliner_threshold=args.gliner_threshold,
        cache_dir=args.cache_dir,
        device=device,
        batch_size=args.batch_size,
    )

    per_cat = collections.defaultdict(collections.Counter)
    label_confusion = collections.defaultdict(collections.Counter)
    survive_before = collections.Counter()
    survive_after = collections.Counter()
    missed_examples = collections.defaultdict(list)
    n_gold = 0

    B = args.batch_size
    for i in range(0, len(notes), B):
        batch_notes = notes[i:i + B]
        batch_pids = pids[i:i + B]
        texts, golds = [], []
        for note in batch_notes:
            t, gsp = inject(note, templates, rng, args.injections_per_note)
            texts.append(t)
            golds.append(gsp)

        plans = deid.plan_batch(texts, batch_pids)

        for text, gsp, plan in zip(texts, golds, plans):
            masked_lower = None
            for g in gsp:
                n_gold += 1
                outcome, label = score(g, plan, text)
                per_cat[g[2]][outcome] += 1
                if outcome != "missed":
                    label_confusion[g[2]][label] += 1
                elif len(missed_examples[g[2]]) < 5:
                    missed_examples[g[2]].append(text[g[0]:g[1]])
            # Clinical-term preservation, measured on the original note only.
            # Some generated names are themselves clinical words ("Candida" is
            # a given name), and redacting an injected one is correct, not a
            # false positive -- so occurrences inside gold spans are excluded
            # from the "before" count.
            from deid2.spans import mask
            after = mask(text, plan).lower()
            before = text.lower()
            injected = " ".join(text[a:b] for a, b, _ in gsp).lower()
            for term, rx in TERM_RX.items():
                b = len(rx.findall(before)) - len(rx.findall(injected))
                if b > 0:
                    survive_before[term] += b
                    survive_after[term] += len(rx.findall(after))
        if (i // B) % 20 == 0:
            print(f"\r  {i + len(batch_notes):,}/{len(notes):,} notes, "
                  f"{n_gold:,} gold spans", end="", file=sys.stderr)
    print(file=sys.stderr)

    lines = ["PHI INJECTION VALIDATION",
             "=" * 74,
             f"notes            : {len(notes):,}",
             f"injections/note  : {args.injections_per_note}",
             f"gold PHI spans   : {n_gold:,}",
             f"NER layer        : {'off' if args.no_ner else 'on'}",
             "",
             "RECALL BY CATEGORY (span-exact)",
             "-" * 74,
             f"  {'category':<10s} {'gold':>8s} {'full':>8s} {'partial':>8s} "
             f"{'missed':>8s} {'recall':>9s} {'any-cover':>10s}"]
    tot = collections.Counter()
    for cat in sorted(per_cat):
        c = per_cat[cat]
        g = sum(c.values())
        tot.update(c)
        rec = 100 * c["full"] / g if g else 0.0
        anyc = 100 * (c["full"] + c["partial"]) / g if g else 0.0
        lines.append(f"  {cat:<10s} {g:>8,} {c['full']:>8,} {c['partial']:>8,} "
                     f"{c['missed']:>8,} {rec:>8.2f}% {anyc:>9.2f}%")
    gt = sum(tot.values())
    lines.append(f"  {'ALL':<10s} {gt:>8,} {tot['full']:>8,} {tot['partial']:>8,} "
                 f"{tot['missed']:>8,} {100*tot['full']/max(gt,1):>8.2f}% "
                 f"{100*(tot['full']+tot['partial'])/max(gt,1):>9.2f}%")

    lines += ["", "CLINICAL TERM PRESERVATION (any loss is a false positive)",
              "-" * 74]
    lost = [(t, survive_before[t], survive_after[t]) for t in survive_before
            if survive_after[t] < survive_before[t]]
    total_b = sum(survive_before.values())
    total_a = sum(survive_after.values())
    lines.append(f"  tracked term occurrences: {total_b:,} before, {total_a:,} after "
                 f"({100*(total_b-total_a)/max(total_b,1):.4f}% lost)")
    if lost:
        for t, b, a in sorted(lost, key=lambda x: x[1] - x[2], reverse=True)[:20]:
            lines.append(f"    {t:<22s} {b:>8,} -> {a:>8,}   (-{b-a})")
    else:
        lines.append("    no tracked clinical term was ever redacted")

    if any(missed_examples.values()):
        lines += ["", "MISSED EXAMPLES", "-" * 74]
        for cat, exs in missed_examples.items():
            if exs:
                lines.append(f"  [{cat}] " + " | ".join(repr(e) for e in exs[:4]))

    report = "\n".join(lines)
    print(report)
    if args.out:
        pathlib.Path(args.out).write_text(report + "\n")


if __name__ == "__main__":
    main()

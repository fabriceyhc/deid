#!/usr/bin/env python3
"""Second-pass de-identification of the provider notes corpus.

The delivered notes arrive one row per line-chunk of each note. PHI spans
straddle those boundaries and the NER layer needs surrounding context, so notes
are reconstructed before de-identification and the resulting spans are mapped
back onto the original lines. Both the line-level file and the reconstructed
note-level file are written, so downstream consumers can use either.

Work is sharded across GPUs by patient id. The notes file is grouped by patient,
so this keeps every line of a note inside one worker.

    python3 run_deid_notes.py \
        --input-csv  .../original/Provider_Notes_deid.csv \
        --identifiers-csv .../original/Patient_Identifiers.csv \
        --out-dir    .../processed/deid2 \
        --gpus 3,4,6
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

NOTE_KEY = ["IP_PATIENT_ID", "IP_ENC_ID", "IP_NOTE_ID"]
JOIN = " "  # matches reconstruct_full_notes() in the project's chunker


def _hash_shard(value: str, n: int) -> int:
    return int(hashlib.md5(str(value).encode()).hexdigest(), 16) % n


def _pseudonymize_author(value: str) -> str:
    """Replace "U0036150 - LAST, FIRST" with a stable non-identifying token.

    The authoring provider stays usable as a grouping key without carrying the
    name, which the upstream extract left in the clear.
    """
    if not isinstance(value, str) or not value.strip():
        return value
    digest = hashlib.sha256(value.strip().encode()).hexdigest()[:10]
    return f"PROV_{digest}"


def _iter_note_groups(input_csv: str, shard: int, n_shards: int,
                      chunksize: int, limit: Optional[int]):
    """Yield complete notes assigned to this shard, in file order.

    Rows of one note may span chunk boundaries, so the tail of each chunk is
    held back until the next chunk confirms the note has ended.
    """
    carry: Optional[pd.DataFrame] = None
    emitted = 0
    for chunk in pd.read_csv(input_csv, chunksize=chunksize, low_memory=False):
        chunk = chunk.copy()
        if n_shards > 1:
            keep = chunk["IP_PATIENT_ID"].map(lambda v: _hash_shard(v, n_shards) == shard)
            chunk = chunk[keep]
        if chunk.empty:
            continue
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last_key = tuple(chunk.iloc[-1][k] for k in NOTE_KEY)
        is_last = pd.Series(True, index=chunk.index)
        for k, v in zip(NOTE_KEY, last_key):
            is_last &= chunk[k] == v
        carry = chunk[is_last]
        ready = chunk[~is_last]
        for _, group in ready.groupby(NOTE_KEY, sort=False):
            yield group
            emitted += 1
            if limit and emitted >= limit:
                return
    if carry is not None and not carry.empty:
        for _, group in carry.groupby(NOTE_KEY, sort=False):
            yield group
            emitted += 1
            if limit and emitted >= limit:
                return


def _reconstruct(group: pd.DataFrame) -> Tuple[str, List[Tuple[int, int]]]:
    """Join a note's lines and return the text plus each line's char range."""
    group = group.sort_values("LINE_NUMBER")
    texts = ["" if pd.isna(t) else str(t) for t in group["NOTE_TEXT"]]
    offsets, pos = [], 0
    for i, t in enumerate(texts):
        if i:
            pos += len(JOIN)
        offsets.append((pos, pos + len(t)))
        pos += len(t)
    return JOIN.join(texts), offsets


def _mask_lines(full_text: str, spans, offsets) -> List[str]:
    """Apply the note-level span plan to each original line."""
    from deid2.spans import mask, Span
    out = []
    for lo, hi in offsets:
        local = [Span(max(s.start, lo) - lo, min(s.end, hi) - lo, s.label,
                      s.source, s.score, s.detail, s.vetoable)
                 for s in spans if s.start < hi and s.end > lo]
        out.append(mask(full_text[lo:hi], local))
    return out


def _worker(args_tuple) -> Dict:
    (args, shard, n_shards, gpu) = args_tuple
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    from deid2 import pipeline
    from deid2.spans import ResolveConfig, mask

    cfg = ResolveConfig()
    if args.keep_dates:
        cfg.enabled_labels.discard("DATE")
    if args.redact_hospitals:
        cfg.enabled_labels.add("HOSP")

    deid = pipeline.build(
        identifiers_csv=args.identifiers_csv,
        roster_path=args.roster,
        use_ner=not args.no_ner,
        ner_models=args.ner_models.split(",") if args.ner_models else None,
        cache_dir=args.cache_dir,
        device="cuda:0" if gpu is not None else "cpu",
        batch_size=args.ner_batch_size,
        config=cfg,
    )

    out_dir = pathlib.Path(args.out_dir)
    line_path = out_dir / f".shard{shard}_lines.csv"
    full_path = out_dir / f".shard{shard}_full.csv"
    line_first = full_first = True

    buf_groups: List[pd.DataFrame] = []
    buf_texts: List[str] = []
    buf_offsets: List[List[Tuple[int, int]]] = []
    buf_pids: List[str] = []
    n_notes = 0
    t0 = time.time()

    def flush() -> None:
        nonlocal line_first, full_first, buf_groups, buf_texts, buf_offsets, buf_pids
        if not buf_groups:
            return
        plans = deid.plan_batch(buf_texts, buf_pids)
        line_rows, full_rows = [], []
        for group, text, offsets, plan in zip(buf_groups, buf_texts, buf_offsets, plans):
            masked_lines = _mask_lines(text, plan, offsets)
            g = group.sort_values("LINE_NUMBER").copy()
            g["NOTE_TEXT"] = masked_lines
            if "CREATE_BY" in g.columns:
                g["CREATE_BY"] = g["CREATE_BY"].map(_pseudonymize_author)
            line_rows.append(g)

            head = g.iloc[0]
            row = {k: head[k] for k in g.columns if k not in ("LINE_NUMBER", "NOTE_TEXT")}
            # Build the note-level text by joining the masked lines rather than
            # masking the full text again. Masking twice is not idempotent at
            # the seams: the tag-collapse pass merges an adjacent "[NAME]
            # [NAME]" pair in the full text, but cannot when the pair straddles
            # a line boundary, so the two files would disagree on ~1% of notes
            # and joining the line file would not reproduce the note file.
            row["NOTE_TEXT"] = JOIN.join(masked_lines)
            full_rows.append(row)

        pd.concat(line_rows, ignore_index=True).to_csv(
            line_path, mode="w" if line_first else "a", header=line_first, index=False)
        pd.DataFrame(full_rows).to_csv(
            full_path, mode="w" if full_first else "a", header=full_first, index=False)
        line_first = full_first = False
        buf_groups, buf_texts, buf_offsets, buf_pids = [], [], [], []

    for group in _iter_note_groups(args.input_csv, shard, n_shards,
                                   args.chunk_size, args.limit):
        text, offsets = _reconstruct(group)
        buf_groups.append(group)
        buf_texts.append(text)
        buf_offsets.append(offsets)
        buf_pids.append(group.iloc[0]["IP_PATIENT_ID"])
        n_notes += 1
        if len(buf_groups) >= args.note_batch_size:
            flush()
            if shard == 0:
                rate = n_notes / max(time.time() - t0, 1e-9)
                print(f"\r  shard0: {n_notes:,} notes  ({rate:.1f} notes/s)",
                      end="", file=sys.stderr)
    flush()
    if shard == 0:
        print(file=sys.stderr)

    return {"shard": shard, "notes": n_notes,
            "stats": deid.stats.render() if deid.stats else "",
            "by_label": dict(deid.stats.spans_by_label) if deid.stats else {},
            "by_source": dict(deid.stats.spans_by_source) if deid.stats else {}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--identifiers-csv", default=None,
                    help="Patient_Identifiers.csv; enables the self-referential layer")
    ap.add_argument("--roster", default=None,
                    help="Provider roster file (built from CREATE_BY if absent)")
    ap.add_argument("--gpus", default="",
                    help="Comma-separated GPU ids, e.g. 3,4,6. Empty means CPU.")
    ap.add_argument("--cache-dir", default="/local1/fabricehc/huggingface/hub")
    ap.add_argument("--ner-models", default=None)
    ap.add_argument("--no-ner", action="store_true")
    ap.add_argument("--ner-batch-size", type=int, default=64)
    ap.add_argument("--note-batch-size", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=50_000)
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after this many notes per shard (for smoke tests)")
    ap.add_argument("--keep-dates", action="store_true",
                    help="Leave dates intact (for a limited data set)")
    ap.add_argument("--redact-hospitals", action="store_true")
    ap.add_argument("--prefix", default="Provider_Notes")
    ap.add_argument("--force-gpus", action="store_true",
                    help="Use the requested GPUs even if another job holds memory there")
    ap.add_argument("--gpu-free-threshold-mib", type=int, default=2000)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpus: List[Optional[int]] = [int(g) for g in args.gpus.split(",") if g.strip()] or [None]
    if not args.force_gpus:
        from deid2.gpu import check_gpus_free
        check_gpus_free(gpus, args.gpu_free_threshold_mib)
    n_shards = len(gpus)

    # Build the roster once up front so the workers do not race to write it.
    if args.roster and not pathlib.Path(args.roster).exists():
        from deid2.gazetteer import RosterGazetteer
        print("Building provider roster from CREATE_BY...", file=sys.stderr)
        RosterGazetteer.from_notes(args.input_csv).save(args.roster)

    print(f"Sharding across {n_shards} worker(s) on GPUs {gpus}", file=sys.stderr)
    t0 = time.time()
    payload = [(args, i, n_shards, gpus[i]) for i in range(n_shards)]
    if n_shards == 1:
        results = [_worker(payload[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_shards) as pool:
            results = pool.map(_worker, payload)

    # Merge shards, restoring the original file order.
    for kind, out_name in (("lines", f"{args.prefix}_deid2.csv"),
                           ("full", f"{args.prefix}_full_deid2.csv")):
        shards = [out_dir / f".shard{r['shard']}_{kind}.csv" for r in results]
        shards = [p for p in shards if p.exists()]
        if not shards:
            continue
        if len(shards) == 1:
            # One shard is already in file order; renaming avoids reading and
            # rewriting the whole corpus for nothing.
            shards[0].rename(out_dir / out_name)
            print(f"wrote {out_dir / out_name}", file=sys.stderr)
            continue
        frames = [pd.read_csv(p, low_memory=False) for p in shards]
        merged = pd.concat(frames, ignore_index=True)
        sort_cols = [c for c in NOTE_KEY + ["LINE_NUMBER"] if c in merged.columns]
        merged = merged.sort_values(sort_cols, kind="mergesort")
        merged.to_csv(out_dir / out_name, index=False)
        for p in shards:
            p.unlink()
        print(f"wrote {out_dir / out_name}  ({len(merged):,} rows)", file=sys.stderr)

    total_notes = sum(r["notes"] for r in results)
    by_label: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    for r in results:
        for k, v in r["by_label"].items():
            by_label[k] = by_label.get(k, 0) + v
        for k, v in r["by_source"].items():
            by_source[k] = by_source.get(k, 0) + v

    lines = [f"notes processed : {total_notes:,}",
             f"elapsed         : {time.time() - t0:.0f}s",
             f"spans redacted  : {sum(by_label.values()):,}",
             "  by category:"]
    lines += [f"    {k:<10s} {v:>10,}" for k, v in sorted(by_label.items(), key=lambda kv: -kv[1])]
    lines += ["  by layer:"]
    lines += [f"    {k:<10s} {v:>10,}" for k, v in sorted(by_source.items(), key=lambda kv: -kv[1])]
    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "deid2_run_report.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()

"""Transformer token-classification layer.

Runs one or more clinical de-identification NER models over arbitrarily long
notes using overlapping windows, and returns character spans in the original
text. Several models can be stacked; their spans are unioned, because recall is
what this layer exists to provide -- precision is restored downstream by the
allowlist veto in ``spans.resolve``.

Note the default model choice. The original pipeline used
``StanfordAIMI/stanford-deidentifier-only-i2b2``, which scored 38.6% recall in
this repo's own benchmark; ``obi/deid_roberta_i2b2`` is the stronger checkpoint
and distinguishes PATIENT from STAFF, which the arbitration layer uses.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .spans import Span

DEFAULT_MODELS = ("obi/deid_roberta_i2b2",)

# Model label -> canonical category. OTHERPHI is intentionally unmapped: it is
# the noisiest class and drives most false positives.
LABEL_MAP: Dict[str, str] = {
    "PATIENT": "NAME",
    "STAFF": "NAME",
    "DOCTOR": "NAME",
    "PATIENT_NAME": "NAME",
    "NAME": "NAME",
    "HCW": "NAME",
    "DATE": "DATE",
    "AGE": "AGE",
    "PHONE": "PHONE",
    "FAX": "PHONE",
    "EMAIL": "EMAIL",
    "ID": "ID",
    "MEDICALRECORD": "ID",
    "IDNUM": "ID",
    "DEVICE": "ID",
    "LOC": "LOC",
    "LOCATION": "LOC",
    "CITY": "CITY",
    "STATE": "LOC",
    "STREET": "ADDRESS",
    "ZIP": "ZIP",
    "HOSP": "HOSP",
    "HOSPITAL": "HOSP",
    "PATORG": "HOSP",
    "ORGANIZATION": "HOSP",
    "URL": "URL",
}

_PREFIXES = ("B-", "I-", "L-", "U-", "E-", "S-")


def _strip_prefix(label: str) -> tuple:
    for p in _PREFIXES:
        if label.startswith(p):
            return label[len(p):], p
    return label, ""


class TransformerNERMasker:
    """Batched, windowed NER over long clinical notes."""

    def __init__(
        self,
        models: Sequence[str] = DEFAULT_MODELS,
        cache_dir: Optional[str] = None,
        device: str = "cuda:0",
        max_length: int = 512,
        stride: int = 128,
        batch_size: int = 64,
        fp16: bool = True,
    ):
        self.device = device
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size
        self._loaded = []
        dtype = torch.float16 if (fp16 and "cuda" in device) else torch.float32
        for name in models:
            # A checkpoint that fails to load is fatal on purpose. Continuing
            # with fewer models than requested would silently reduce coverage
            # while the caller still believes the full ensemble ran -- the
            # worst failure mode for a de-identification pass. HuggingFace has
            # published "deid" repos whose weights are a text placeholder, so
            # this is a real scenario, not a hypothetical one.
            try:
                tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir,
                                                    add_prefix_space=True)
                mod = AutoModelForTokenClassification.from_pretrained(
                    name, cache_dir=cache_dir, dtype=dtype).to(device).eval()
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load NER checkpoint {name!r}: "
                    f"{type(exc).__name__}: {exc}\n"
                    f"Refusing to run a de-identification pass with an "
                    f"incomplete model set. Verify the checkpoint (a stub "
                    f"repo will have a tiny model.safetensors) or drop it "
                    f"from --ner-models."
                ) from exc
            n_labels = len(getattr(mod.config, "id2label", {}) or {})
            if n_labels < 2:
                raise RuntimeError(
                    f"Checkpoint {name!r} declares {n_labels} labels; that is "
                    f"not a usable token-classification model.")
            self._loaded.append((name, tok, mod))

    def get_spans_batch(self, texts: Sequence[str]) -> List[List[Span]]:
        """Return one span list per input text (union across loaded models)."""
        results: List[List[Span]] = [[] for _ in texts]
        for name, tok, mod in self._loaded:
            for i, spans in enumerate(self._run_model(texts, tok, mod)):
                results[i].extend(spans)
        return results

    def get_spans(self, text: str) -> List[Span]:
        return self.get_spans_batch([text])[0]

    def _run_model(self, texts: Sequence[str], tok, mod) -> List[List[Span]]:
        out: List[List[Span]] = [[] for _ in texts]
        # Encode everything into overlapping windows, tracking which text each
        # window came from so spans can be mapped back.
        enc = tok(
            list(texts),
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )
        sample_map = enc["overflow_to_sample_mapping"]
        n_windows = len(enc["input_ids"])
        id2label = mod.config.id2label

        order = sorted(range(n_windows), key=lambda i: len(enc["input_ids"][i]))
        for bstart in range(0, n_windows, self.batch_size):
            idxs = order[bstart:bstart + self.batch_size]
            batch = tok.pad(
                {"input_ids": [enc["input_ids"][i] for i in idxs],
                 "attention_mask": [enc["attention_mask"][i] for i in idxs]},
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = mod(**batch).logits
            probs = torch.softmax(logits.float(), dim=-1)
            conf, pred = probs.max(dim=-1)
            conf = conf.cpu().numpy()
            pred = pred.cpu().numpy()

            for row, widx in enumerate(idxs):
                offsets = enc["offset_mapping"][widx]
                mask = enc["attention_mask"][widx]
                labels = [id2label[int(pred[row, j])] for j in range(len(offsets))]
                scores = [float(conf[row, j]) for j in range(len(offsets))]
                out[sample_map[widx]].extend(
                    _decode(labels, scores, offsets, mask))
        return out


def _decode(labels: Sequence[str], scores: Sequence[float],
            offsets: Sequence[Sequence[int]], mask: Sequence[int]) -> List[Span]:
    """Turn per-token tags into character spans, merging adjacent same-type tags."""
    spans: List[Span] = []
    cur_cat = None
    cur_start = cur_end = 0
    cur_scores: List[float] = []

    def flush() -> None:
        nonlocal cur_cat, cur_scores
        if cur_cat is not None and cur_end > cur_start:
            spans.append(Span(cur_start, cur_end, cur_cat, source="ner",
                              score=sum(cur_scores) / len(cur_scores)))
        cur_cat, cur_scores = None, []

    for j, raw in enumerate(labels):
        if j >= len(mask) or not mask[j]:
            continue
        start, end = offsets[j]
        if start == end:  # special token
            continue
        base, _ = _strip_prefix(raw)
        cat = LABEL_MAP.get(base.upper()) if base.upper() != "O" else None
        if cat is None:
            flush()
            continue
        # Same category and contiguous (or separated only by whitespace) --
        # extend rather than start a new span.
        if cat == cur_cat and start - cur_end <= 1:
            cur_end = end
            cur_scores.append(scores[j])
        else:
            flush()
            cur_cat, cur_start, cur_end, cur_scores = cat, start, end, [scores[j]]
    flush()
    return spans

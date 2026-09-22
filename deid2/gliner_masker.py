"""GLiNER layer.

GLiNER is a different architecture from the BIO token classifiers in ner.py: a
bi-encoder that scores arbitrary natural-language entity labels against spans,
so the label set is supplied at inference time rather than baked into a
classification head. That makes it worth testing here -- the PHI categories can
be described in words rather than mapped from whatever tag set a checkpoint
happened to be trained on.

It has a hard input limit and no built-in windowing, so long notes are split
into overlapping character windows and spans are mapped back.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .spans import Span

# Natural-language labels -> canonical categories.
DEFAULT_LABELS: Dict[str, str] = {
    "person name": "NAME",
    "patient name": "NAME",
    "doctor name": "NAME",
    "family member name": "NAME",
    "date": "DATE",
    "date of birth": "DATE",
    "age over 89": "AGE",
    "phone number": "PHONE",
    "email address": "EMAIL",
    "medical record number": "ID",
    "identification number": "ID",
    "street address": "ADDRESS",
    "city": "CITY",
    "zip code": "ZIP",
    "hospital name": "HOSP",
    "url": "URL",
}


class GLiNERMasker:
    """Windowed GLiNER inference returning character spans."""

    def __init__(
        self,
        model_name: str = "Lekhansh/modern-gliner-large-mednotes-deidentifier",
        cache_dir: Optional[str] = None,
        device: str = "cuda:0",
        labels: Optional[Dict[str, str]] = None,
        threshold: float = 0.5,
        window_chars: int = 1500,
        overlap_chars: int = 200,
        batch_size: int = 8,
    ):
        from gliner import GLiNER

        self.model = GLiNER.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = self.model.to(device).eval()
        self.label_map = labels or DEFAULT_LABELS
        self.labels = list(self.label_map)
        self.threshold = threshold
        self.window_chars = window_chars
        self.overlap_chars = overlap_chars
        self.batch_size = batch_size

    def _windows(self, text: str):
        step = max(1, self.window_chars - self.overlap_chars)
        for start in range(0, max(1, len(text)), step):
            chunk = text[start:start + self.window_chars]
            if chunk:
                yield start, chunk

    def get_spans_batch(self, texts: Sequence[str]) -> List[List[Span]]:
        # Flatten every window of every text into one list, run them in
        # batches, then scatter the results back to their source text.
        flat, owner, offsets = [], [], []
        for i, t in enumerate(texts):
            for off, chunk in self._windows(t):
                flat.append(chunk)
                owner.append(i)
                offsets.append(off)

        out: List[List[Span]] = [[] for _ in texts]
        for b in range(0, len(flat), self.batch_size):
            chunk_batch = flat[b:b + self.batch_size]
            try:
                preds = self.model.batch_predict_entities(
                    chunk_batch, self.labels, threshold=self.threshold)
            except Exception:
                preds = [self.model.predict_entities(c, self.labels,
                                                     threshold=self.threshold)
                         for c in chunk_batch]
            for j, ents in enumerate(preds):
                idx = b + j
                base, who = offsets[idx], owner[idx]
                for e in ents:
                    cat = self.label_map.get(str(e.get("label", "")).lower())
                    if not cat:
                        continue
                    out[who].append(Span(base + int(e["start"]),
                                         base + int(e["end"]),
                                         cat, source="ner",
                                         score=float(e.get("score", 1.0))))
        return out

    def get_spans(self, text: str) -> List[Span]:
        return self.get_spans_batch([text])[0]

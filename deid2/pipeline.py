"""Orchestration: run every layer over a batch of texts and produce masked output.

Layer order is precision-descending. The gazetteer layers know the answer from
the structured tables and are trusted unconditionally; the rule layer carries
contextual anchors; the NER layer supplies recall and is filtered by the
allowlist. ``spans.resolve`` performs the arbitration.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

from .gazetteer import PatientGazetteer, RosterGazetteer
from .rules import ClinicalRuleMasker
from .spans import ResolveConfig, Span, mask, resolve

RESOURCE_DIR = pathlib.Path(__file__).parent / "resources"


def load_allowlists(resource_dir: Optional[str] = None):
    d = pathlib.Path(resource_dir or RESOURCE_DIR)
    def _read(name: str) -> Set[str]:
        p = d / name
        return set(p.read_text().split()) if p.exists() else set()
    return _read("clinical_allowlist.txt"), _read("general_allowlist.txt")


class CompositeNER:
    """Runs several NER backends and concatenates their spans.

    Recall is the point; precision is restored by the allowlist veto in
    spans.resolve, so backends with different blind spots compose cleanly.
    """

    def __init__(self, backends):
        self.backends = list(backends)

    def get_spans_batch(self, texts):
        out = [[] for _ in texts]
        for b in self.backends:
            for i, spans in enumerate(b.get_spans_batch(list(texts))):
                out[i].extend(spans)
        return out

    def get_spans(self, text):
        return self.get_spans_batch([text])[0]


@dataclass
class DeidStats:
    texts: int = 0
    spans_by_label: Dict[str, int] = field(default_factory=dict)
    spans_by_source: Dict[str, int] = field(default_factory=dict)

    def add(self, spans: Sequence[Span]) -> None:
        self.texts += 1
        for s in spans:
            self.spans_by_label[s.label] = self.spans_by_label.get(s.label, 0) + 1
            self.spans_by_source[s.source] = self.spans_by_source.get(s.source, 0) + 1

    def render(self) -> str:
        lines = [f"texts processed : {self.texts:,}"]
        total = sum(self.spans_by_label.values())
        lines.append(f"spans redacted  : {total:,}")
        if self.texts:
            lines.append(f"spans per text  : {total / self.texts:.2f}")
        lines.append("  by category:")
        for k, v in sorted(self.spans_by_label.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {k:<10s} {v:>10,}")
        lines.append("  by layer:")
        for k, v in sorted(self.spans_by_source.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {k:<10s} {v:>10,}")
        return "\n".join(lines)


class Deidentifier2:
    """Multi-layer second-pass de-identifier."""

    def __init__(
        self,
        patient_gazetteer: Optional[PatientGazetteer] = None,
        roster: Optional[RosterGazetteer] = None,
        rules: Optional[ClinicalRuleMasker] = None,
        ner=None,
        config: Optional[ResolveConfig] = None,
        resource_dir: Optional[str] = None,
        collect_stats: bool = True,
    ):
        self.patient_gazetteer = patient_gazetteer
        self.roster = roster
        self.rules = rules
        self.ner = ner
        self.config = config or ResolveConfig()
        self.clinical_allow, self.general_allow = load_allowlists(resource_dir)
        self.stats = DeidStats() if collect_stats else None

    def plan_batch(self, texts: Sequence[str],
                   patient_ids: Optional[Sequence[Optional[str]]] = None) -> List[List[Span]]:
        """Return the resolved, non-overlapping span plan for each text."""
        pids = list(patient_ids) if patient_ids is not None else [None] * len(texts)
        raw: List[List[Span]] = [[] for _ in texts]

        for i, text in enumerate(texts):
            if self.patient_gazetteer is not None:
                raw[i].extend(self.patient_gazetteer.get_spans(text, pids[i]))
            if self.roster is not None:
                raw[i].extend(self.roster.get_spans(text))
            if self.rules is not None:
                raw[i].extend(self.rules.get_spans(text))

        if self.ner is not None:
            for i, spans in enumerate(self.ner.get_spans_batch(list(texts))):
                raw[i].extend(spans)

        plans = []
        for text, spans in zip(texts, raw):
            plan = resolve(text, spans, self.clinical_allow,
                           self.general_allow, self.config)
            if self.stats is not None:
                self.stats.add(plan)
            plans.append(plan)
        return plans

    def deidentify_batch(self, texts: Sequence[str],
                         patient_ids: Optional[Sequence[Optional[str]]] = None) -> List[str]:
        plans = self.plan_batch(texts, patient_ids)
        return [mask(t, p) for t, p in zip(texts, plans)]

    def deidentify(self, text: str, patient_id: Optional[str] = None) -> str:
        return self.deidentify_batch([text], [patient_id])[0]


def build(
    identifiers_csv: Optional[str] = None,
    roster_path: Optional[str] = None,
    notes_csv_for_roster: Optional[str] = None,
    use_ner: bool = True,
    ner_models: Optional[Sequence[str]] = None,
    gliner_model: Optional[str] = None,
    gliner_threshold: float = 0.5,
    cache_dir: Optional[str] = None,
    device: str = "cuda:0",
    batch_size: int = 64,
    config: Optional[ResolveConfig] = None,
    resource_dir: Optional[str] = None,
) -> Deidentifier2:
    """Assemble a de-identifier from the study's own files."""
    pg = PatientGazetteer(identifiers_csv) if identifiers_csv else None

    clinical_allow, general_allow = load_allowlists(resource_dir)
    ambiguous = general_allow | clinical_allow

    roster = None
    if roster_path and pathlib.Path(roster_path).exists():
        roster = RosterGazetteer.load(roster_path, ambiguous=ambiguous)
    elif notes_csv_for_roster:
        roster = RosterGazetteer.from_notes(notes_csv_for_roster)
        roster.ambiguous = {a.lower() for a in ambiguous}
        if roster_path:
            roster.save(roster_path)

    ner = None
    if use_ner:
        backends = []
        if ner_models is None or list(ner_models) != ["none"]:
            from .ner import DEFAULT_MODELS, TransformerNERMasker
            backends.append(TransformerNERMasker(
                models=ner_models or DEFAULT_MODELS,
                cache_dir=cache_dir, device=device, batch_size=batch_size))
        if gliner_model:
            from .gliner_masker import GLiNERMasker
            backends.append(GLiNERMasker(
                model_name=gliner_model, cache_dir=cache_dir, device=device,
                threshold=gliner_threshold, batch_size=max(4, batch_size // 4)))
        ner = backends[0] if len(backends) == 1 else CompositeNER(backends)

    return Deidentifier2(patient_gazetteer=pg, roster=roster,
                         rules=ClinicalRuleMasker(), ner=ner,
                         config=config, resource_dir=resource_dir)

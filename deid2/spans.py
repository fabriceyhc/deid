"""Span representation, cross-layer arbitration, and typed masking.

The de-identification layers (gazetteer, rules, NER) each emit ``Span`` objects
independently. ``resolve`` fuses them into a single non-overlapping sequence,
applying the allowlist veto and source-priority rules, and ``mask`` renders the
result with typed tags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

# Source layers, most trustworthy first. A span produced by a higher-priority
# layer wins territory disputes and cannot be vetoed by the allowlist.
SOURCE_PRIORITY: Dict[str, int] = {
    "self": 100,     # the patient's own identifiers, from the structured tables
    "roster": 80,    # provider names harvested from CREATE_BY
    "rule": 60,      # label-anchored and format regexes
    "ner": 40,       # transformer token classification
}

# Canonical PHI categories -> the tag written into the output text.
TAGS: Dict[str, str] = {
    "NAME": "[NAME]",
    "DATE": "[DATE]",
    "AGE": "[AGE]",
    "PHONE": "[PHONE]",
    "EMAIL": "[EMAIL]",
    "ID": "[ID]",
    "ADDRESS": "[ADDRESS]",
    "CITY": "[CITY]",
    "ZIP": "[ZIP]",
    "LOC": "[LOC]",
    "HOSP": "[HOSP]",
    "URL": "[URL]",
}

# The self-referential layer is never vetoed: if a token is genuinely this
# patient's name it gets redacted even when it collides with a clinical term.
NON_VETOABLE_SOURCES: Set[str] = {"self"}

_WORD = re.compile(r"[A-Za-z]+")
_EXISTING_MASK = re.compile(r"_{3,}|\[[A-Z]+\]")
_DIGIT = re.compile(r"\d")
_WORDCHAR = re.compile(r"[A-Za-z0-9]")
_FRAGMENT_NUM = re.compile(r"\d{1,2}")
_BARE_YEAR = re.compile(r"(?:19|20)\d{2}")
# Digits plus only the separators a phone number may legitimately contain.
_PHONE_SHAPE = re.compile(r"[\d\s().+\-]+")


def _snap(text: str, span: Span) -> Span:
    """Grow a span to whole-word edges.

    Windowed NER cuts entities at window boundaries, leaving partial spans that
    redact "RON" out of "RONALD" and leave the tail exposed.
    """
    start, end = span.start, span.end
    n = len(text)
    # A degenerate or out-of-range span would index past the ends below.
    if end <= start or start < 0 or end > n:
        return span
    while start > 0 and _WORDCHAR.match(text[start - 1]) and _WORDCHAR.match(text[start]):
        start -= 1
    while end < n and _WORDCHAR.match(text[end]) and _WORDCHAR.match(text[end - 1]):
        end += 1
    if (start, end) == (span.start, span.end):
        return span
    return Span(start, end, span.label, span.source, span.score,
                span.detail, span.vetoable)


def _plausible(label: str, surface: str, cfg: "ResolveConfig") -> bool:
    """Reject detections whose surface form cannot be an instance of the label.

    Applied to the NER layer only. It emits sub-token fragments at window edges
    -- a stray "14" tagged PHONE, a "16" tagged DATE -- and these checks remove
    the bulk of that noise. The rule layer is exempt because its patterns
    already encode the format: a four-digit pager extension matched behind the
    word "pager" is a real one, and would be thrown away by the digit-count
    heuristics below.
    """
    text = surface.strip()
    if not text:
        return False
    digits = len(_DIGIT.findall(text))
    # A span containing an upstream "___" has run across a field boundary --
    # typically a row of vital-sign timestamps, "___ 1539 ___ 1542", which has
    # enough digits to look like a phone number but is not one.
    spans_a_mask = "___" in text
    if label == "PHONE":
        if spans_a_mask or not _PHONE_SHAPE.fullmatch(text):
            return False
        return 7 <= digits <= 15
    if label == "ID":
        if spans_a_mask:
            return False
        # A bare four-digit number is a room, extension or department code far
        # more often than a record number.
        if text.isdigit():
            return digits >= 5
        return digits >= 4
    if label == "ZIP":
        return not spans_a_mask and digits >= 5
    if label == "DATE":
        # Order timestamps ("___ 1620") are bare digit runs the NER layer likes
        # to call dates. A digit run is only a date if it is a plausible year.
        if text.isdigit():
            if _BARE_YEAR.fullmatch(text):
                return cfg.redact_bare_years
            return False
        return True
    if label in ("NAME", "ADDRESS", "LOC", "CITY", "HOSP"):
        return len(text) >= 2 and bool(_WORD.search(text))
    return True


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    source: str = "rule"
    score: float = 1.0
    detail: str = ""
    # Anchored detections ("Dr. X", "PATIENT: X") carry their own contextual
    # evidence and are exempt from the allowlist veto.
    vetoable: bool = True

    @property
    def priority(self) -> int:
        return SOURCE_PRIORITY.get(self.source, 0)

    def __len__(self) -> int:
        return self.end - self.start


@dataclass
class ResolveConfig:
    """Which categories to act on, and how permissive to be."""

    enabled_labels: Set[str] = field(
        default_factory=lambda: {
            "NAME", "DATE", "PHONE", "EMAIL", "ID",
            "ADDRESS", "CITY", "ZIP", "LOC", "URL", "AGE",
        }
    )
    # Safe Harbor only requires ages 90+ to be suppressed; redacting every age
    # would strip clinically load-bearing information for no privacy gain.
    min_redacted_age: int = 90
    # Minimum NER score to accept a span from the transformer layer alone.
    ner_threshold: float = 0.5
    # Above this score, a transformer span outranks the general-English
    # allowlist (but not the clinical one).
    strong_evidence: float = 0.90
    # A bare NER span this short is almost always a tokenizer artifact.
    min_span_chars: int = 2
    # A four-digit year on its own is permitted under Safe Harbor and carries
    # real clinical meaning ("Dx of AIDS in 2019"), so it is kept by default.
    redact_bare_years: bool = False


def _span_text(text: str, span: Span) -> str:
    return text[span.start : span.end]


def _is_allowlisted(surface: str, allowlist: Set[str]) -> bool:
    """True when every alphabetic token of the surface form is a known clinical term.

    Requiring *all* tokens to be benign keeps "Kaposi sarcoma" safe while still
    redacting "Kaposi Hernandez", where only one token is clinical vocabulary.
    """
    tokens = _WORD.findall(surface.lower())
    if not tokens:
        return False
    return all(t in allowlist for t in tokens)


def _age_value(surface: str) -> Optional[int]:
    m = re.search(r"\d{1,3}", surface)
    return int(m.group()) if m else None


def resolve(
    text: str,
    spans: Iterable[Span],
    allowlist: Optional[Set[str]] = None,
    weak_allowlist: Optional[Set[str]] = None,
    config: Optional[ResolveConfig] = None,
) -> List[Span]:
    """Filter, arbitrate and flatten raw spans into a masking plan.

    Two allowlist tiers are applied. ``allowlist`` holds clinical vocabulary
    (drug names, eponyms, lab components, note headers) and vetoes any vetoable
    span. ``weak_allowlist`` holds general English and only vetoes weak
    evidence, so a confident "Dr. Baker" survives while the word "baker" in
    running prose does not get redacted.
    """
    cfg = config or ResolveConfig()
    allow = allowlist or set()
    weak = weak_allowlist or set()

    kept: List[Span] = []
    for sp in spans:
        if sp.label not in cfg.enabled_labels:
            continue
        sp = _snap(text, sp)
        surface = _span_text(text, sp)
        if not surface.strip():
            continue

        # Never re-redact something that is already a mask.
        if _EXISTING_MASK.fullmatch(surface.strip()):
            continue

        if sp.source == "ner" and not _plausible(sp.label, surface, cfg):
            continue

        if sp.source == "ner":
            if sp.score < cfg.ner_threshold:
                continue
            if len(surface.strip()) < cfg.min_span_chars:
                continue

        if sp.label == "AGE":
            value = _age_value(surface)
            if value is None or value < cfg.min_redacted_age:
                continue

        if sp.vetoable and sp.source not in NON_VETOABLE_SOURCES:
            if _is_allowlisted(surface, allow):
                continue
            if sp.score < cfg.strong_evidence and _is_allowlisted(surface, weak):
                continue

        kept.append(sp)

    return _flatten(kept)


def _flatten(spans: Sequence[Span]) -> List[Span]:
    """Collapse overlaps into a non-overlapping, position-sorted sequence.

    Overlapping spans are merged into their union; the label and source come
    from the highest-priority (then longest) contributor, so a self-referential
    NAME hit overlapping a vague NER LOC hit is reported as NAME.
    """
    if not spans:
        return []

    ordered = sorted(spans, key=lambda s: (s.start, -s.end))
    out: List[Span] = []
    cur = ordered[0]
    members = [cur]

    for sp in ordered[1:]:
        # Overlapping, or touching and of the same kind -- a windowed NER pass
        # routinely splits one entity across two adjacent spans. The adjacency
        # rule is restricted to that layer so it cannot chain spans from
        # different layers into one runaway redaction.
        touching_ner = (sp.label == cur.label and sp.source == "ner"
                        and cur.source == "ner" and sp.start - cur.end <= 1)
        if sp.start < cur.end or touching_ner:
            members.append(sp)
            cur = Span(
                start=cur.start,
                end=max(cur.end, sp.end),
                label=cur.label,
                source=cur.source,
                score=cur.score,
            )
        else:
            out.append(_pick(cur, members))
            cur = sp
            members = [sp]
    out.append(_pick(cur, members))
    return out


def _pick(union: Span, members: Sequence[Span]) -> Span:
    best = max(members, key=lambda s: (s.priority, len(s), s.score))
    return Span(
        start=union.start,
        end=union.end,
        label=best.label,
        source=best.source,
        score=best.score,
        detail=best.detail,
        vetoable=best.vetoable,
    )


# Collapse a run of the *same* tag, which happens when several layers each
# catch one token of one name. Upstream "___" markers are deliberately left
# alone: absorbing them would erase the record that a separate item was
# redacted there by the first pass.
# The possessive quantifier is load-bearing -- a plain `[\s,]*` here backtracks
# exponentially on the long "______" form separators these notes are full of.
_COLLAPSE = re.compile(r"(\[[A-Z]+\])(?:[\s,]*+\1)+")


def mask(text: str, spans: Sequence[Span], collapse: bool = True) -> str:
    """Replace resolved spans with their typed tags."""
    if not spans:
        return text
    parts: List[str] = []
    last = 0
    for sp in spans:
        if sp.start < last:
            continue
        parts.append(text[last : sp.start])
        parts.append(TAGS.get(sp.label, "[REDACTED]"))
        last = sp.end
    parts.append(text[last:])
    out = "".join(parts)
    if collapse:
        # "[NAME] [NAME]" -> "[NAME]"; keeps the text from ballooning where
        # several layers each caught one token of the same name.
        out = _COLLAPSE.sub(r"\1", out)
    return out

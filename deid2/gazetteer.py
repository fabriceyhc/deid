"""Self-referential and roster-based matching.

This is the layer that closes the dominant leak in the upstream de-identified
notes: the patient's own name, in the clear, in their own chart. Because the
study already holds a patient -> name/DOB crosswalk in the structured tables, a
patient identifier does not have to be *inferred* from the text at all -- it can
be looked up and matched exactly. That makes this layer both the highest-recall
and the highest-precision one available.
"""

from __future__ import annotations

import datetime as _dt
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .spans import Span

try:
    import ahocorasick
except ImportError:  # pragma: no cover - the roster layer degrades to regex
    ahocorasick = None

_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_BOUNDARY = re.compile(r"[A-Za-z0-9_]")

# Common English given-name hypocorisms. Charts routinely record the legal name
# while the narrative uses the familiar form, so matching the registry string
# alone misses a large share of real occurrences.
NICKNAMES: Dict[str, Sequence[str]] = {
    "robert": ("bob", "bobby", "rob", "robbie", "bert"),
    "richard": ("rick", "ricky", "dick", "rich", "richie"),
    "william": ("will", "bill", "billy", "willie", "liam"),
    "james": ("jim", "jimmy", "jamie"),
    "john": ("jon", "johnny", "jack"),
    "jonathan": ("jon", "john", "johnny"),
    "joseph": ("joe", "joey"),
    "michael": ("mike", "mikey", "mick"),
    "charles": ("charlie", "chuck", "chas"),
    "thomas": ("tom", "tommy"),
    "christopher": ("chris", "topher"),
    "daniel": ("dan", "danny"),
    "matthew": ("matt", "matty"),
    "anthony": ("tony",),
    "donald": ("don", "donnie"),
    "steven": ("steve", "stevie"),
    "stephen": ("steve", "stevie"),
    "andrew": ("andy", "drew"),
    "kenneth": ("ken", "kenny"),
    "edward": ("ed", "eddie", "ted", "teddy"),
    "timothy": ("tim", "timmy"),
    "devrim": ("jeff",),
    "gregory": ("greg",),
    "benjamin": ("ben", "benny"),
    "samuel": ("sam", "sammy"),
    "alexander": ("alex", "alec", "xander"),
    "nicholas": ("nick", "nicky"),
    "patrick": ("pat", "paddy"),
    "raymond": ("ray",),
    "lawrence": ("larry",),
    "frederick": ("fred", "freddie"),
    "theodore": ("ted", "teddy", "theo"),
    "albert": ("al", "bert"),
    "eugene": ("gene",),
    "ronald": ("ron", "ronnie"),
    "douglas": ("doug",),
    "peter": ("pete",),
    "elizabeth": ("liz", "beth", "betty", "eliza", "lizzie", "betsy"),
    "margaret": ("maggie", "peggy", "meg", "marge"),
    "katherine": ("kate", "katie", "kathy", "kat"),
    "catherine": ("cathy", "kate", "katie", "cate"),
    "patricia": ("pat", "patty", "tricia", "trish"),
    "jennifer": ("jen", "jenny"),
    "deborah": ("deb", "debbie"),
    "barbara": ("barb", "babs"),
    "susan": ("sue", "susie", "suzy"),
    "jessica": ("jess", "jessie"),
    "sarah": ("sally",),
    "rebecca": ("becky", "becca"),
    "victoria": ("vicky", "vicki", "tori"),
    "cynthia": ("cindy",),
    "dorothy": ("dot", "dottie"),
    "virginia": ("ginny", "ginger"),
    "theresa": ("terry", "tess", "tracy"),
    "veronica": ("ronnie", "vera"),
    "pamela": ("pam",),
    "christina": ("chris", "tina", "christy"),
    "kimberly": ("kim",),
    "eleanor": ("ellie", "nell"),
    "abigail": ("abby",),
    "penelope": ("penny",),
    "gabriel": ("gabe",),
    "manuel": ("manny",),
    "guadalupe": ("lupe",),
    "francisco": ("paco", "frank", "cisco"),
    "jose": ("pepe",),
    "antonio": ("tony",),
    "alejandro": ("alex",),
}


def _variants(token: str) -> Set[str]:
    """Surface forms a single name token may take in narrative text."""
    tok = token.strip().lower()
    out = {tok}
    out.update(NICKNAMES.get(tok, ()))
    # Hyphenated and apostrophed surnames get split, mangled, or half-redacted.
    for piece in re.split(r"[-']", tok):
        if len(piece) >= 3:
            out.add(piece)
    return {v for v in out if len(v) >= 3}


def parse_name(raw: str) -> Tuple[List[str], List[str]]:
    """Split a registry name string into (surnames, given names).

    Accepts both "LAST, FIRST MIDDLE" and "FIRST MIDDLE LAST".
    """
    raw = str(raw or "").strip()
    if not raw:
        return [], []
    if "," in raw:
        last, _, first = raw.partition(",")
        return _TOKEN.findall(last), _TOKEN.findall(first)
    toks = _TOKEN.findall(raw)
    if len(toks) == 1:
        return toks, []
    return toks[-1:], toks[:-1]


def _dob_patterns(dob: str) -> List[str]:
    """Regex alternatives for a date of birth as it may be written in prose."""
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S", "%m/%d/%y", "%d-%b-%Y"):
        try:
            d = _dt.datetime.strptime(str(dob).strip(), fmt).date()
            break
        except (ValueError, TypeError):
            continue
    else:
        return []
    month_name = d.strftime("%B")
    month_abbr = d.strftime("%b")
    return [
        rf"{d.month}/{d.day}/{d.year}", rf"{d.month:02d}/{d.day:02d}/{d.year}",
        rf"{d.month}-{d.day}-{d.year}", rf"{d.month:02d}-{d.day:02d}-{d.year}",
        rf"{d.year}-{d.month:02d}-{d.day:02d}",
        rf"{d.month}/{d.day}/{str(d.year)[2:]}",
        rf"{month_name}\s+{d.day}(?:st|nd|rd|th)?,?\s+{d.year}",
        rf"{month_abbr}\.?\s+{d.day}(?:st|nd|rd|th)?,?\s+{d.year}",
    ]


class PatientGazetteer:
    """Matches a patient's own registered identifiers inside their own notes."""

    def __init__(self, identifiers_csv: str, max_cache: int = 4096):
        df = pd.read_csv(identifiers_csv)
        self._names: Dict[str, str] = {}
        self._dobs: Dict[str, str] = {}
        for row in df.itertuples(index=False):
            pid = getattr(row, "IP_PATIENT_ID")
            self._names[pid] = str(getattr(row, "PAT_NAME", "") or "")
            self._dobs[pid] = str(getattr(row, "DOB", "") or "")
        self._compile = lru_cache(maxsize=max_cache)(self._compile_uncached)

    def __len__(self) -> int:
        return len(self._names)

    def _compile_uncached(self, pid: str):
        surnames, givens = parse_name(self._names.get(pid, ""))
        sur_v, giv_v = set(), set()
        for s in surnames:
            sur_v |= _variants(s)
        for g in givens:
            giv_v |= _variants(g)

        patterns: List[Tuple[re.Pattern, str, float]] = []

        # Full name in either order scores highest and is matched first so it
        # is redacted as one span rather than two adjacent ones.
        if sur_v and giv_v:
            g = "|".join(sorted(map(re.escape, giv_v), key=len, reverse=True))
            s = "|".join(sorted(map(re.escape, sur_v), key=len, reverse=True))
            patterns.append((re.compile(
                rf"\b(?:{g})\s+(?:[A-Z]\.?\s+)?(?:{s})\b", re.I), "NAME", 1.0))
            patterns.append((re.compile(rf"\b(?:{s}),\s*(?:{g})\b", re.I), "NAME", 1.0))

        for v in sorted(sur_v | giv_v, key=len, reverse=True):
            patterns.append((re.compile(rf"\b{re.escape(v)}(?:'s)?\b", re.I), "NAME", 1.0))

        for dob_pat in _dob_patterns(self._dobs.get(pid, "")):
            patterns.append((re.compile(rf"\b{dob_pat}\b", re.I), "DATE", 1.0))

        return tuple(patterns)

    def get_spans(self, text: str, patient_id: Optional[str]) -> List[Span]:
        if not patient_id or patient_id not in self._names:
            return []
        spans: List[Span] = []
        for pattern, label, score in self._compile(patient_id):
            for m in pattern.finditer(text):
                spans.append(Span(m.start(), m.end(), label,
                                  source="self", score=score, vetoable=False))
        return spans


class RosterGazetteer:
    """Matches provider names harvested from the note metadata itself.

    ``CREATE_BY`` carries an authored-by string per note line, so the full staff
    roster for the corpus can be recovered from the data with no external list.
    """

    def __init__(self, surnames: Iterable[str], givens: Iterable[str],
                 ambiguous: Optional[Set[str]] = None):
        self.surnames = {s.lower() for s in surnames if len(s) >= 3}
        self.givens = {g.lower() for g in givens if len(g) >= 3}
        # Roster surnames that are also ordinary English words ("Young", "Long",
        # "Bell"). Matching these on sight would redact running prose, so they
        # are emitted with weak evidence and left to the general-English veto in
        # spans.resolve; the anchored rules still catch "Dr. Young".
        self.ambiguous = {a.lower() for a in (ambiguous or set())}
        self._auto = None
        if ahocorasick is not None and self.surnames:
            auto = ahocorasick.Automaton()
            for s in self.surnames:
                auto.add_word(s, s)
            auto.make_automaton()
            self._auto = auto
        self._fallback = None
        if self._auto is None and self.surnames:
            alt = "|".join(sorted(map(re.escape, self.surnames), key=len, reverse=True))
            self._fallback = re.compile(rf"\b(?:{alt})\b", re.I)

    @classmethod
    def from_notes(cls, notes_csv: str, column: str = "CREATE_BY",
                   chunksize: int = 200_000) -> "RosterGazetteer":
        surnames, givens = set(), set()
        for chunk in pd.read_csv(notes_csv, usecols=[column],
                                 chunksize=chunksize, low_memory=False):
            for raw in chunk[column].dropna().unique():
                # "U0036150 - LAST, FIRST MIDDLE"
                name = str(raw).split(" - ", 1)[-1]
                sur, giv = parse_name(name)
                for s in sur:
                    # Terminated-employee records are prefixed "TERM-".
                    s = re.sub(r"^TERM-", "", s, flags=re.I)
                    surnames |= _variants(s)
                for g in giv:
                    givens |= _variants(g)
        return cls(surnames, givens)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            fh.write("# surnames\n")
            fh.write("\n".join(sorted(self.surnames)))
            fh.write("\n# givens\n")
            fh.write("\n".join(sorted(self.givens)) + "\n")

    @classmethod
    def load(cls, path: str, ambiguous: Optional[Set[str]] = None) -> "RosterGazetteer":
        surnames, givens, bucket = [], [], None
        for line in open(path):
            line = line.strip()
            if line == "# surnames":
                bucket = surnames
            elif line == "# givens":
                bucket = givens
            elif line and bucket is not None:
                bucket.append(line)
        return cls(surnames, givens, ambiguous)

    def get_spans(self, text: str) -> List[Span]:
        if not self.surnames:
            return []
        spans: List[Span] = []
        lower = text.lower()
        n = len(lower)

        def emit(start: int, end: int) -> None:
            if start > 0 and _BOUNDARY.match(lower[start - 1]):
                return
            if end < n and _BOUNDARY.match(lower[end]):
                return
            # Staff names are written capitalised; an all-lowercase hit is the
            # common noun, not the person.
            if not text[start].isupper():
                return
            surface = text[start:end]
            # Short all-caps runs are clinical acronyms (POA = present on
            # admission, ROS = review of systems), not the surnames they
            # collide with.
            if surface.isupper() and len(surface) <= 4:
                return
            score = 0.85 if lower[start:end] in self.ambiguous else 1.0
            # Absorb an immediately preceding roster given name, so
            # "Devrim Ferrin" becomes one span instead of a bare surname.
            s = start
            prev = lower[max(0, start - 40):start]
            m = re.search(r"([A-Za-z][A-Za-z'\-]*)(?:\s+[A-Za-z]\.?)?[\s,]+$", prev)
            if m and m.group(1) in self.givens:
                s = start - (len(prev) - m.start(1))
            spans.append(Span(s, end, "NAME", source="roster", score=score))

        if self._auto is not None:
            for end_idx, word in self._auto.iter(lower):
                emit(end_idx - len(word) + 1, end_idx + 1)
        else:
            for m in self._fallback.finditer(lower):
                emit(m.start(), m.end())
        return spans

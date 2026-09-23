"""Label-anchored and format-based rules.

These target the concrete failure modes measured in the upstream-de-identified
corpus rather than trying to be a general PHI regex suite:

  * names surviving next to a redacted MRN/DOB in template headers
    ("PATIENT: Jane Doe  MRN: ___")
  * half-redacted hyphenated surnames ("___-Thornbury, MD R3")
  * credential-suffixed names ("Rosalind E. Ramirez, MSN")
  * discharge street addresses and ZIPs the upstream pass left intact
  * month-name dates, pager/extension numbers, ages over 89

Anchored patterns carry their own contextual evidence and are marked
non-vetoable, so the allowlist cannot suppress them.
"""

from __future__ import annotations

import re
from typing import List, Optional, Pattern, Set, Tuple

from .spans import Span

# A capitalised name-shaped token. Excludes ALL-CAPS runs, which in these notes
# are almost always section headers or abbreviations.
NAME_TOK = (r"(?:O'|D'|Mc|Mac|Van\s|Von\s|De\s|La\s|Le\s)?"
            r"[A-Z][a-z]{1,20}(?:[A-Z][a-z]{1,20})?(?:['\u2019\-][A-Z]?[a-z]{1,20})?")
# The middle-initial alternative must not swallow the first letter of the next
# name token, so it requires a non-letter to follow.
# Notes arrive as one long line with multi-space field separators, so the gap
# between tokens of one name is capped tightly and the match is trimmed of
# template vocabulary afterwards by _trim_name.
NAME_SEQ = rf"{NAME_TOK}(?:[ \t]{{1,3}}(?:{NAME_TOK}|[A-Z]\.?(?![A-Za-z]))){{0,3}}"

CREDENTIALS = (r"M\.?D\.?|D\.?O\.?|N\.?P\.?|P\.?A\.?-?C?|R\.?N\.?|MSN|BSN|CNS|CNM|CRNA|"
               r"L\.?V\.?N\.?|Ph\.?D\.?|Psy\.?D\.?|D\.?D\.?S\.?|D\.?P\.?M\.?|R\.?D\.?|"
               r"RDN|CNSC|LCSW|MSW|PharmD|R\.?Ph\.?|PGY-?\d|R\d|MPH")

# Section labels after which a name-shaped token is a real name, not prose.
# "Med Name:", "Test Name:", "Drug Name:" label a *thing*, not a person.
# Without these lookbehinds the bare "Name" anchor captures the medication:
# "Med Name: Morphine Clonodine Pain" was redacted to "[NAME]".
NAME_ANCHORS = (r"Patient(?:\s+Name)?|Pt\.?\s*Name|"
                r"(?<!Med\s)(?<!Medication\s)(?<!Drug\s)(?<!Test\s)(?<!Lab\s)"
                r"(?<!Order\s)(?<!Product\s)(?<!Device\s)Name|"
                r"PATIENT|Resident|Intern|Attending|"
                r"Fellow|Provider|Physician(?:\s+ordering)?|Surgeon|Consultant|Referring|"
                r"Author|Signed\s+by|Electronically\s+signed(?:\s+by)?|Dictated\s+by|"
                r"Cosigned(?:\s+by)?|Entered\s+by|Ordered\s+by|Performed\s+by|"
                r"Contact\s+Name|Emergency\s+Contact|Next\s+of\s+Kin|Guarantor|"
                r"Primary\s+Care(?:\s+Physician)?|PCP|Discharged?\s+by")

# A personal title is an unambiguous person marker, unlike the generic field
# labels above, so it gets its own handle for the veto exception below.
TITLE_ANCHOR = re.compile(
    rf"\b(?:Dr|Doctor|Mr|Mrs|Ms|Miss|Prof)\.?[ \t]+({NAME_SEQ})\b")

# (pattern, label, group, vetoable)
_RULES: List[Tuple[Pattern, str, int, bool]] = [
    # ---- names with contextual anchors -------------------------------------
    # Labels are often compounded with slashes: "Pt. Name/Age/DOB:".
    (re.compile(rf"\b(?:{NAME_ANCHORS})(?:[ \t]*/[ \t]*[A-Za-z.]+)*"
                rf"[ \t]*[:\-][ \t]*({NAME_SEQ})"), "NAME", 1, False),
    (TITLE_ANCHOR, "NAME", 1, False),
    (re.compile(rf"\b({NAME_SEQ})\s*,\s*(?:{CREDENTIALS})\b"), "NAME", 1, False),
    (re.compile(rf"\b({NAME_SEQ})\s+(?:{CREDENTIALS})\b(?!\s*[a-z])"), "NAME", 1, False),
    # "Surname,Given" is how contact and code-status fields are written
    # ("Primary contact: Danvers,Marlowe K"). The NER layer reliably tags the
    # surname and just as reliably misses the given name after the comma,
    # leaving a relative's first name in the clear.
    #
    # Only the anchored form is matched: an adjacent mask is proof a name was
    # there. A bare "Word, Word" pattern is far too broad in clinical text --
    # it matches drug lists ("Latuda, Risperdal"), tox panels ("Amphetamines,
    # Barbituates") and specimen types ("Blood, Peripheral").
    # Half-redacted hyphenated surnames left behind by the upstream pass.
    (re.compile(r"_{3,}\s*-\s*([A-Z][a-z]{1,20})\b"), "NAME", 1, False),
    (re.compile(rf"\b({NAME_TOK})\s*-\s*_{{3,}}"), "NAME", 1, False),
    # "Briefly, Mr. X is a 48 y.o." -- narrative openers.
    (re.compile(rf"\b({NAME_SEQ})\s+is\s+an?\s+\d{{1,3}}\s*(?:y/?o|yo|year[\s-]old|F\b|M\b)"),
     "NAME", 1, True),

    # ---- identifiers --------------------------------------------------------
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "ID", 0, False),
    (re.compile(r"\b(?:MRN|MR#|Medical\s+Record(?:\s+Number)?|Account|Acct)\s*[:#]?\s*(\d{4,})\b",
                re.I), "ID", 1, False),
    (re.compile(r"\b(?:pager|pgr|beeper|ext(?:ension)?)\s*[:#]?\s*(\d{3,6})\b", re.I),
     "PHONE", 1, False),
    (re.compile(r"(?<=[,\s])x\s?(\d{4,6})\b"), "PHONE", 1, False),
    (re.compile(r"(?<![\d.\-/])(?:\+?1[-.\s]?)?(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]?\d{4}(?![\d-])"),
     "PHONE", 0, False),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "EMAIL", 0, False),

    # ---- geography ----------------------------------------------------------
    (re.compile(r"\b\d{1,5}\s+(?:[NSEW]\.?\s+)?(?:[A-Z][A-Za-z]*\s+){0,4}"
                r"(?:St|Street|Ave|Avenue|Blvd|Boulevard|Rd|Road|Dr|Drive|Ln|Lane|Ct|Court|"
                r"Way|Pl|Place|Ter|Terrace|Pkwy|Parkway|Hwy|Highway|Cir|Circle|Trl|Trail)\b\.?"
                r"(?:\s*(?:#|Apt\.?|Suite|Ste\.?|Unit|Rm\.?|Room)\s*[\w\-]+)?"
                r"(?:\s*,?\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}\s*,?\s*"
                r"(?:CA|California|AZ|NV|OR|WA|TX|NY|FL)\b\.?"
                r"(?:\s*,?\s*\d{5}(?:-\d{4})?)?)?"), "ADDRESS", 0, False),
    # City + state + ZIP as one unit. The street-address rule only absorbs the
    # locality when it is comma-joined ("... Ave, Glendale CA 91203"); written
    # as "... Blvd in Glendale CA 91203" the city would otherwise survive, and
    # a city is a Safe Harbor identifier in its own right.
    (re.compile(r"\b([A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+){0,2}[,\s]+"
                r"(?:CA|California|AZ|NV|OR|WA|TX|NY|FL)[,\s]+\d{5}(?:-\d{4})?)\b"),
     "CITY", 1, False),
    (re.compile(r"\b(?:CA|California|AZ|NV|OR|WA|TX|NY|FL)[,\s]+(\d{5}(?:-\d{4})?)\b"),
     "ZIP", 1, False),
    (re.compile(r"\b(?:zip|postal)\s*(?:code)?\s*[:#]?\s*(\d{5}(?:-\d{4})?)\b", re.I),
     "ZIP", 1, False),

    # ---- dates and ages -----------------------------------------------------
    (re.compile(r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)?\d{2}\b"),
     "DATE", 0, False),
    (re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+"
                r"\d{1,2}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{2}\b", re.I), "DATE", 0, False),
    (re.compile(r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+"
                r"(?:19|20)\d{2}\b", re.I), "DATE", 0, False),
    # Ages are filtered to 90+ downstream by ResolveConfig.min_redacted_age.
    # "y.o." with the periods is the dominant form in these notes, so the
    # separator inside the abbreviation has to be optional too.
    (re.compile(r"\b\d{2,3}[\s-]*(?:y[./\s]?o\.?|years?[\s-]*old|yrs?\.?[\s-]*old)",
                re.I), "AGE", 0, False),

    # ---- web ----------------------------------------------------------------
    (re.compile(r"\bhttps?://\S+"), "URL", 0, False),
    (re.compile(r"\bwww\.[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\S*"), "URL", 0, False),
]

# Template vocabulary that a greedy name match may absorb from an adjacent
# field. Trimmed off either end of a candidate name span.
_HEADER_WORDS = set("""
patient pt name names physician provider attending resident intern fellow student
surgeon consultant referring author signed cosigned dictated entered ordered
performed discharged admitted contact emergency kin guarantor primary secondary
care doctor nurse nursing therapist pharmacist dietitian social worker team
service department unit room bed floor clinic hospital center medical health
date time dob birth age sex gender race mrn record number account phone fax
address city state zip email history physical assessment plan impression
recommendation subjective objective chief complaint present illness past
surgical family review systems allergies medications vitals labs imaging
discharge admission summary course disposition followup follow signature
none unknown other see same self deferred pending refused declined yes no
male female home multiple various per with without and the for from
to at in on by of or as is was be do due new old per via than then
todo tbd pcp rn np pa md do dds rd lvn bsn msn
preference preferences instruction instructions choice choices option options
list listing details detail info information section form template field
""".split())

_NAME_TOKEN_SPLIT = re.compile(r"[ \t]+")
_FOLLOWED_BY_COLON = re.compile(r"[ \t]*:")

# Template headers put the patient's name immediately before a (redacted) MRN
# or DOB field: "PATIENT:  Alden Prescott  MRN: ___". Anchoring on the field
# label and looking back is far cheaper than scanning a name pattern at every
# offset and rejecting it with a lookahead.
_ID_FIELD = re.compile(r"\b(?:MRN|MR#|DOB|D\.O\.B\.?|Date\s+of\s+Birth)\b", re.I)

# A given name surviving after a masked surname: "Danvers,Marlowe K" becomes
# "[NAME],MARLOWE K" when the NER tags the surname and misses the rest.
# Ungated, this pattern is overwhelmingly credentials -- "Jane Smith, PA-C"
# leaves "[NAME], PA-C" -- so on a 2,000-note sample it fired 405 times and
# only 3 were names. It is therefore gated on a known-given-name lookup, which
# on the same sample keeps 12 hits, all of them real.
_MASK_THEN_WORD = re.compile(
    r"(?:\[NAME\]|_{3,})[,;][ \t]?([A-Z][A-Za-z'\-]{1,20}(?:[ \t]+[A-Z]\b)?)")

CREDENTIAL_TOKENS = set("""
pa pac ms ma rn np md do pt ot rd cc ct ca dr phy sp od sh aud cpo facc lmft
mbbs asw ms4 msw lcsw crna dpm dds bsn msn dnp acnp fnp agacnp rdap aahivs
cnm cns rdn cnsc lvn rph pharmd psyd phd edd mph mba faap facp fccp
""".split())
_NAME_BEFORE_FIELD = re.compile(rf"({NAME_SEQ})[ \t]+$")


def _names_before_id_fields(text: str):
    """Yield (start, end) of name-shaped runs sitting just before an MRN/DOB label."""
    for m in _ID_FIELD.finditer(text):
        window_start = max(0, m.start() - 60)
        window = text[window_start:m.start()]
        hit = _NAME_BEFORE_FIELD.search(window)
        if hit:
            yield window_start + hit.start(1), window_start + hit.end(1)


def _trim_name(text: str, start: int, end: int):
    """Shrink a candidate name span so it excludes adjacent template words."""
    surface = text[start:end]
    # Offsets of each whitespace-separated token within the surface.
    toks, pos = [], 0
    for piece in _NAME_TOKEN_SPLIT.split(surface):
        idx = surface.index(piece, pos) if piece else pos
        toks.append((idx, idx + len(piece), piece))
        pos = idx + len(piece)
    toks = [t for t in toks if t[2]]
    while toks and toks[0][2].lower().strip(".,'-") in _HEADER_WORDS:
        toks.pop(0)
    while toks and toks[-1][2].lower().strip(".,'-") in _HEADER_WORDS:
        toks.pop()
    if not toks:
        return None
    return start + toks[0][0], start + toks[-1][1]


# Token sequences that look like anchored names but are template furniture.
_HEADER_NOISE = re.compile(
    r"^(?:None|Unknown|N/?A|Not\s|No\s|See\s|Same|Self|Patient|Pt\b|Deferred|Pending|"
    r"Refused|Declined|Yes|No|Male|Female|Home|Other|Multiple|Various|Per\s)",
    re.I)


class ClinicalRuleMasker:
    """Regex layer tuned to the residual-PHI profile of this corpus."""

    def __init__(self, redact_urls: bool = True,
                 given_names: Optional[Set[str]] = None,
                 clinical_allowlist: Optional[Set[str]] = None,
                 general_allowlist: Optional[Set[str]] = None):
        self.redact_urls = redact_urls
        self.given_names = {g.lower() for g in (given_names or set())}
        self.clinical_allowlist = {c.lower() for c in (clinical_allowlist or set())}
        self.general_allowlist = {g.lower() for g in (general_allowlist or set())}

    def _given_name_after_mask(self, text: str) -> List[Span]:
        """Given names left behind after a masked surname, gated on a name list."""
        if not self.given_names:
            return []
        out: List[Span] = []
        for m in _MASK_THEN_WORD.finditer(text):
            word = m.group(1).split()[0]
            key = word.lower().replace("-", "").replace("'", "")
            if len(word) < 4:
                continue
            if key in CREDENTIAL_TOKENS or key in self.clinical_allowlist:
                continue
            if key not in self.given_names:
                # An unusual given name will not be in any name list
                # ("MARLOWE", "MARYKE"). Accept a long all-caps token as a
                # fallback, but only unhyphenated: every credential that
                # reaches here is hyphenated ("GNP-BC", "ACNP-BC", "NSCA-CPT")
                # as is the one facility form ("UCLA-SM").
                if not (word.isupper() and len(word) >= 6
                        and "-" not in word
                        and key not in self.general_allowlist):
                    continue
            out.append(Span(m.start(1), m.end(1), "NAME",
                            source="rule", score=1.0, vetoable=False))
        return out

    def get_spans(self, text: str) -> List[Span]:
        spans: List[Span] = self._given_name_after_mask(text)
        for start, end in _names_before_id_fields(text):
            trimmed = _trim_name(text, start, end)
            if trimmed is not None:
                spans.append(Span(trimmed[0], trimmed[1], "NAME",
                                  source="rule", score=1.0, vetoable=False))
        for pattern, label, group, vetoable in _RULES:
            if label == "URL" and not self.redact_urls:
                continue
            for m in pattern.finditer(text):
                score = 1.0
                start, end = m.span(group)
                if start < 0:
                    continue
                surface = text[start:end]
                if not surface.strip():
                    continue
                if label == "NAME":
                    if _HEADER_NOISE.match(surface.strip()):
                        continue
                    trimmed = _trim_name(text, start, end)
                    if trimmed is None:
                        continue
                    start, end = trimmed
                    # A one-word "name" behind an anchor is often template
                    # vocabulary; let the allowlist have a say. Multi-token
                    # names keep their anchor's authority.
                    if len(_NAME_TOKEN_SPLIT.split(text[start:end].strip())) == 1:
                        vetoable, score = True, 0.85
                        # Exception: a personal title is unambiguous. "Dr. Read"
                        # is a person even though "read" is a dictionary word.
                        # The giveaway for a compound field label -- "Doctor
                        # Preferences:" -- is the colon that follows it.
                        if (pattern is TITLE_ANCHOR
                                and not _FOLLOWED_BY_COLON.match(text, end)):
                            vetoable, score = False, 1.0
                    else:
                        score = 1.0
                spans.append(Span(start, end, label, source="rule",
                                  score=score, vetoable=vetoable))
        return spans

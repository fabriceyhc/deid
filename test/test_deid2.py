"""Tests for the second-pass de-identification layers.

Run with:  python3 -m pytest test/test_deid2.py -q
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from deid2.gazetteer import RosterGazetteer, _dob_patterns, _variants, parse_name
from deid2.rules import ClinicalRuleMasker
from deid2.spans import ResolveConfig, Span, mask, resolve

CLINICAL = {"kaposi", "sarcoma", "vancomycin", "sepsis", "cellulitis", "july", "may"}
GENERAL = {"baker", "preferences", "league", "young", "morning", "meals"}


def deid(text, spans, cfg=None):
    return mask(text, resolve(text, spans, CLINICAL, GENERAL, cfg))


GIVEN = {"robin", "ruben", "jacqueline", "chellee", "gary", "susan", "emma"}


def rule_deid(text, cfg=None, given=GIVEN):
    masker = ClinicalRuleMasker(given_names=given, clinical_allowlist=CLINICAL)
    return deid(text, masker.get_spans(text), cfg)


# --------------------------------------------------------------------------
# name parsing and variant generation
# --------------------------------------------------------------------------

def test_parse_name_both_orders():
    assert parse_name("SMITH, ROBERT JAMES") == (["SMITH"], ["ROBERT", "JAMES"])
    assert parse_name("Robert James Smith") == (["Smith"], ["Robert", "James"])


def test_nickname_variants_are_tuples_not_strings():
    # A bare string in the nickname table would be splatted into characters.
    assert _variants("Anthony") == {"anthony", "tony"}
    assert "t" not in _variants("Anthony")


def test_hyphenated_surname_yields_both_halves():
    assert {"smith", "jones"} <= _variants("Smith-Jones")


def test_dob_patterns_cover_common_formats():
    pats = _dob_patterns("1970-03-14")
    assert "3/14/1970" in pats
    assert any("March" in p for p in pats)


# --------------------------------------------------------------------------
# allowlist arbitration
# --------------------------------------------------------------------------

def test_clinical_term_survives_confident_ner():
    t = "Kaposi sarcoma noted."
    spans = [Span(0, 14, "NAME", "ner", 0.99)]
    assert deid(t, spans) == t


def test_patient_own_name_is_never_vetoed():
    # "may" is clinical vocabulary, but the self layer outranks the allowlist.
    t = "Patient May reports pain."
    spans = [Span(8, 11, "NAME", "self", 1.0, vetoable=False)]
    assert deid(t, spans) == "Patient [NAME] reports pain."


def test_general_word_vetoes_only_weak_evidence():
    t = "the baker delivered bread"
    assert deid(t, [Span(4, 9, "NAME", "ner", 0.6)]) == t
    # A confident detection outranks the general tier.
    assert "[NAME]" in deid(t, [Span(4, 9, "NAME", "ner", 0.95)])


# --------------------------------------------------------------------------
# implausible-span filtering
# --------------------------------------------------------------------------

@pytest.mark.parametrize("surface,label", [
    ("1620", "DATE"),   # an order timestamp, not a date
    ("14", "PHONE"),    # a window-edge fragment
    ("1011", "ID"),     # a room or extension code
])
def test_implausible_spans_are_dropped(surface, label):
    t = f"value {surface} here"
    spans = [Span(6, 6 + len(surface), label, "ner", 0.9)]
    assert deid(t, spans) == t


def test_bare_year_kept_by_default_and_redactable_on_request():
    t = "Dx of AIDS in 2019."
    spans = [Span(14, 18, "DATE", "ner", 0.9)]
    assert deid(t, spans) == t
    cfg = ResolveConfig(redact_bare_years=True)
    assert deid(t, [Span(14, 18, "DATE", "ner", 0.9)], cfg) == "Dx of AIDS in [DATE]."


def test_age_redacted_only_at_ninety_and_above():
    t = "a 94 y.o. male and a 48 y.o. female"
    out = deid(t, [Span(2, 10, "AGE", "ner", 0.9), Span(21, 29, "AGE", "ner", 0.9)])
    assert "[AGE]" in out and "48 y.o." in out


# --------------------------------------------------------------------------
# span geometry
# --------------------------------------------------------------------------

def test_span_snaps_to_word_boundary():
    # A windowed NER pass can cut "RONALD" after "RON".
    t = "sent to UCLA RONALD REAGAN"
    out = deid(t, [Span(13, 16, "LOC", "ner", 0.95)])
    assert "RONALD" not in out


def test_adjacent_ner_spans_merge_but_layers_do_not_chain():
    t = "LOS ANGELES"
    out = deid(t, [Span(0, 3, "LOC", "ner", 0.9), Span(4, 11, "LOC", "ner", 0.9)])
    assert out == "[LOC]"


def test_mask_collapse_is_linear_on_underscore_runs():
    # A non-possessive quantifier here backtracks exponentially.
    import time
    t = "Note: " + "_" * 400 + " [NAME]"
    start = time.time()
    mask(t, [])
    assert time.time() - start < 1.0


# --------------------------------------------------------------------------
# rule layer
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("PATIENT: Jeremy Johnson  MRN: ___", "PATIENT: [NAME]  MRN: ___"),
    ("Pt. Name/Age/DOB:  Patricia Anglano   51 y.o.",
     "Pt. Name/Age/DOB:  [NAME]   51 y.o."),
    ("Attending: Robert J. McDonald-Vance, MD", "Attending: [NAME], MD"),
    ("Ms. Sabina McDonald is a 94 y.o. woman", "Ms. [NAME] is a [AGE] woman"),
    ("___-Prakash, MD R3", "___-[NAME], MD R3"),
    ("Dr. O'Brien agreed", "Dr. [NAME] agreed"),
])
def test_anchored_name_rules(text, expected):
    assert rule_deid(text) == expected


@pytest.mark.parametrize("text", [
    "Outpatient Provider To be determined",
    "Doctor Preferences: none",
    "Home PT: IV League therapy",
])
def test_template_vocabulary_is_not_a_name(text):
    assert rule_deid(text) == text


def test_name_match_does_not_run_into_the_next_field():
    t = "Contact Name: Mary Smith  Physician ordering: Ana Lopez, MSN"
    out = rule_deid(t)
    assert out.count("[NAME]") == 2
    assert "Physician ordering:" in out


def test_address_absorbs_trailing_locality():
    t = "Discharge Address: 660 South Cloverdale Ave. #109, Los Angeles CA 90036"
    assert rule_deid(t) == "Discharge Address: [ADDRESS]"


# --------------------------------------------------------------------------
# roster layer
# --------------------------------------------------------------------------

def test_roster_ignores_short_all_caps_acronyms():
    # "POA" (present on admission) collides with the surname Poa.
    roster = RosterGazetteer(surnames=["poa", "chung"], givens=["jeffrey"])
    assert roster.get_spans("Sepsis POA  Cellulitis") == []


def test_roster_requires_capitalisation():
    roster = RosterGazetteer(surnames=["young"], givens=[])
    assert roster.get_spans("a young male") == []
    assert roster.get_spans("Dr. Young saw him") != []


def test_roster_absorbs_preceding_given_name():
    roster = RosterGazetteer(surnames=["chung"], givens=["jeffrey"])
    t = "seen by Jeffrey Chung today"
    spans = roster.get_spans(t)
    assert t[spans[0].start:spans[0].end] == "Jeffrey Chung"


# --------------------------------------------------------------------------
# regressions from the first full-corpus inspection
# --------------------------------------------------------------------------

def test_medication_sig_is_not_a_name():
    # "Meals" is a real surname on the provider roster; the weak tier has to
    # stop it from eating medication instructions.
    roster = RosterGazetteer(surnames=["meals"], givens=[],
                             ambiguous={"meals"})
    t = "TAKE 3 TABLETS BY MOUTH 3 TIMES A DAY WITH MEALS"
    assert deid(t, roster.get_spans(t)) == t


def test_phone_span_may_not_cross_an_upstream_mask():
    # A row of vital-sign timestamps has enough digits to look like a phone.
    t = "Pain Score Weight   ___ 1539 ___ 1542   36.7 C"
    spans = [Span(20, 37, "PHONE", "ner", 0.9)]
    assert deid(t, spans) == t


@pytest.mark.parametrize("phone", ["310-825-9111", "(310) 825-9111", "+1 310 825 9111"])
def test_real_phone_numbers_still_redacted(phone):
    t = f"call {phone} now"
    spans = [Span(5, 5 + len(phone), "PHONE", "ner", 0.9)]
    assert deid(t, spans) == "call [PHONE] now"


def test_pager_extension_survives_the_digit_heuristics():
    # Rule-sourced spans bypass the NER fragment filters.
    assert rule_deid("Hospitalist  pager 30465") == "Hospitalist  pager [PHONE]"


def test_url_is_not_vetoed_by_the_allowlist():
    # A URL tokenises into ordinary words and was being suppressed.
    out = rule_deid("see www.uclahealth.org/dementia/home")
    assert out == "see [URL]"


def test_upstream_underscore_markers_are_preserved():
    t = "PATIENT: ___ Johnson  seen today"
    out = deid(t, [Span(13, 20, "NAME", "ner", 0.95)])
    assert out == "PATIENT: ___ [NAME]  seen today"


def test_personal_title_overrides_the_dictionary_veto():
    # "read" and "wear" are dictionary words but are also real surnames; a
    # personal title makes them unambiguous.
    for surname in ("Read", "Wear", "Young"):
        t = f"Dr. {surname} saw the patient"
        assert rule_deid(t) == "Dr. [NAME] saw the patient", surname


def test_compound_field_label_is_still_not_a_name():
    # The colon is what distinguishes "Doctor Preferences:" from "Dr. Read".
    assert rule_deid("Doctor Preferences: none") == "Doctor Preferences: none"


def test_city_state_zip_redacted_when_not_comma_joined():
    t = "Home address is 9034 Sunset Blvd in Glendale CA 91203."
    assert rule_deid(t) == "Home address is [ADDRESS] in [CITY]."


def test_line_and_note_output_stay_consistent():
    """Joining the masked lines must reproduce the masked note exactly.

    Masking is not idempotent at line seams: the tag-collapse pass merges an
    adjacent "[NAME] [NAME]" pair inside one string but cannot see across a
    line boundary. The note-level text therefore has to be built from the
    masked lines, not by masking the full text a second time.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rdn", pathlib.Path(__file__).resolve().parents[1] / "run_deid_notes.py")
    rdn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rdn)

    # Two names that collapse when adjacent, split across a line boundary.
    lines = ["Seen by Ana", "Lopez today"]
    full = rdn.JOIN.join(lines)
    offsets, pos = [], 0
    for i, ln in enumerate(lines):
        if i:
            pos += len(rdn.JOIN)
        offsets.append((pos, pos + len(ln)))
        pos += len(ln)

    plan = [Span(8, 11, "NAME", "ner", 0.95), Span(12, 17, "NAME", "ner", 0.95)]
    masked_lines = rdn._mask_lines(full, plan, offsets)
    assert rdn.JOIN.join(masked_lines) == "Seen by [NAME] [NAME] today"
    # Each line is masked independently and covers its own span.
    assert masked_lines == ["Seen by [NAME]", "[NAME] today"]


# --------------------------------------------------------------------------
# comma-joined names (found in the scrubbed corpus, not by unit testing)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("1. [NAME],CHELLEE K Spouse", "1. [NAME] Spouse"),
    ("Contact: [NAME], Robin M  Home Phone:", "Contact: [NAME]  Home Phone:"),
])
def test_given_name_after_a_masked_surname_is_caught(text, expected):
    # "Klepp,Chellee K" -- the NER tags the surname and misses the given name,
    # leaving a relative's first name in the clear.
    assert rule_deid(text) == expected


@pytest.mark.parametrize("text", [
    "Latuda, Risperdal, Zyprexa, Geodon",   # drug list
    "Amphetamines, Barbituates negative",   # tox screen panel
    "Flaxseed, Linseed oil",
    "Blood, Peripheral culture",
    "Calcium, Vitamin D supplement",
])
def test_comma_lists_in_clinical_text_are_not_names(text):
    # An unanchored "Word, Word" rule matches all of these. Redacting a tox
    # panel would silently damage the study's exposure variables, so only the
    # mask-anchored form is allowed.
    assert rule_deid(text) == text


# --------------------------------------------------------------------------
# clinical content preservation (found by the paired before/after audit)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "UNABLE TO FIND Med Name: Morphine Clonodine Pain SQ Pump",
    "Test Name: Candida Auris Surveillance PCR",
    "Drug Name: Suboxone 8-2 mg SL film",
    "Medication Name: buprenorphine-naloxone 8-2 mg",
])
def test_thing_name_labels_are_not_person_anchors(text):
    # "Name:" is a person anchor, but "Med Name:" / "Test Name:" label a thing.
    # Without the lookbehind the medication or lab test was redacted.
    assert rule_deid(text) == text


@pytest.mark.parametrize("text,expect_redacted", [
    ("Patient Name: Jeremy Johnson  MRN: ___", True),
    ("Contact Name: Maria Lopez", True),
    ("Pt. Name/Age/DOB:  Patricia Anglano", True),
])
def test_person_name_anchors_still_fire(text, expect_redacted):
    assert ("[NAME]" in rule_deid(text)) is expect_redacted

#!/usr/bin/env python3
"""Build the clinical + general-English allowlists used to suppress false redactions.

The clinical tier is derived from controlled vocabularies in the study's own
structured tables (medication catalogs, LOINC component names, ICD descriptions,
department names) rather than hand-written, so it tracks the actual corpus.
Free-text columns are deliberately excluded -- they carry patient names.

    python3 tools/build_allowlist.py --data-dir /path/to/original --out deid2/resources
"""

import argparse
import collections
import pathlib
import re
import sys

import pandas as pd

WORD = re.compile(r"[A-Za-z][A-Za-z\-']{1,}")

# (filename, columns) -- controlled vocabularies only.
CATALOG_SOURCES = [
    ("Medications.csv", ["EPIC_MEDICATION_NAME", "MEDISPAN_GENERIC_NAME", "MEDISPAN_CLASS_NAME"]),
    ("Labs_deid.csv", ["COMPONENT_NAME", "PROCEDURE_DESCRIPTION"]),
    ("Problem_Lists_deid.csv", ["ICD_DESCRIPTION"]),
    ("Encounters.csv", ["EPIC_DEPARTMENT_NAME", "DEPARTMENT_SPECIALTY", "IP_VISIT_TYPE",
                        "EPIC_ENCOUNTER_TYPE", "HOSP_DISCHARGE_DISPOSITION", "ED_DISPOSITION"]),
    ("Encounter_Diagnoses.csv", ["ICD_DESCRIPTION"]),
]

# Terms that must never be redacted but are unlikely to appear in the catalogs:
# medical eponyms, note-template headers, and clinical abbreviations that the
# NER models routinely mistake for person names.
CURATED = """
kaposi sarcoma hodgkin non-hodgkin crohn barrett bell palsy graves cushing addison
parkinson alzheimer huntington wilson gaucher fabry pompe tay sachs marfan ehlers danlos
guillain barre charcot marie tooth creutzfeldt jakob wernicke korsakoff broca
hashimoto sjogren behcet wegener churg strauss goodpasture berger buerger raynaud
dupuytren peyronie paget bowen kawasaki still reiter takayasu horton osler weber rendu
sturge klippel trenalunay von willebrand christmas hemophilia glanzmann bernard soulier
evans diamond blackfan fanconi schwachman shwachman kostmann chediak higashi wiskott aldrich
digeorge turner klinefelter down edwards patau prader willi angelman rett
mallory weiss boerhaave zenker meckel hirschsprung whipple zollinger ellison
gilbert dubin prescott rotor crigler najjar budd chiari caroli mirizzi
wilms ewing burkitt waldenstrom richter sezary mycosis fungoides
foley swan ganz jackson pratt penrose malecot pezzer blakemore sengstaken
glasgow ranson apache charlson elixhauser braden morse norton framingham
killip forrester wells geneva centor alvarado curb ottawa nexus canadian
mcburney murphy rovsing psoas obturator homan kernig brudzinski babinski
romberg lasegue phalen tinel finkelstein trendelenburg allen adson
bard parker lister mayo kelly kocher deaver richardson balfour bookwalter
port cath picc midline groshong hickman broviac tenckhoff
auris albicans glabrata tropicalis parapsilosis krusei dubliniensis
aureus epidermidis lugdunensis saprophyticus faecalis faecium
pneumoniae pyogenes agalactiae viridans anginosus constellatus
aeruginosa maltophilia cepacia baumannii cloacae mirabilis marcescens
fragilis nucleatum difficile perfringens tuberculosis avium abscessus
neoformans capsulatum immitis jirovecii fumigatus
lyme legionella salmonella shigella listeria yersinia brucella bartonella rickettsia
klebsiella escherichia serratia proteus providencia morganella citrobacter enterobacter
pseudomonas acinetobacter stenotrophomonas burkholderia moraxella haemophilus
neisseria streptococcus staphylococcus enterococcus clostridium clostridioides
bacteroides fusobacterium prevotella peptostreptococcus actinomyces nocardia
mycobacterium candida aspergillus cryptococcus histoplasma coccidioides blastomyces
pneumocystis toxoplasma giardia entamoeba cryptosporidium strongyloides
gram giemsa wright ziehl neelsen papanicolaou
history physical assessment plan impression recommendation subjective objective
chief complaint present illness past medical surgical family social review systems
allergies medications vitals labs imaging discharge admission summary hospital course
disposition followup follow signature attending resident intern fellow student
patient name date service birth admit discharge provider primary secondary
chart note progress consult operative procedure nursing therapy
male female year old month week day hour minute
morning evening night daily weekly monthly
january february march april may june july august september october november december
monday tuesday wednesday thursday friday saturday sunday
""".split()


def collect_catalog_tokens(data_dir: pathlib.Path, min_count: int) -> set:
    counts = collections.Counter()
    for fname, cols in CATALOG_SOURCES:
        path = data_dir / fname
        if not path.exists():
            print(f"  skip {fname} (not found)", file=sys.stderr)
            continue
        seen_values = set()
        try:
            reader = pd.read_csv(path, usecols=lambda c: c in cols, chunksize=200_000,
                                 low_memory=False)
            rows = 0
            for chunk in reader:
                rows += len(chunk)
                for col in chunk.columns:
                    seen_values.update(chunk[col].dropna().astype(str).unique())
                if rows >= 2_000_000:
                    break
        except Exception as exc:  # a missing column set raises; not fatal
            print(f"  skip {fname}: {exc}", file=sys.stderr)
            continue
        # Count each distinct catalog *value* once, so a drug prescribed a
        # million times does not outweigh the rest.
        for val in seen_values:
            for tok in WORD.findall(val.lower()):
                counts[tok] += 1
        print(f"  {fname}: {len(seen_values)} distinct values", file=sys.stderr)
    return {t for t, c in counts.items() if c >= min_count and len(t) >= 3}


def cohort_name_tokens(notes_csv, ids_csv) -> set:
    """Every name token belonging to a patient or to the provider roster.

    Nothing derived from data may allowlist one of these, or the allowlist would
    end up shielding the very identifiers this pass exists to remove.
    """
    tokens = set()
    if ids_csv and pathlib.Path(ids_csv).exists():
        ids = pd.read_csv(ids_csv)
        for nm in ids.get("PAT_NAME", pd.Series(dtype=str)).dropna().astype(str):
            tokens.update(t.lower() for t in WORD.findall(nm))
    if notes_csv and pathlib.Path(notes_csv).exists():
        for chunk in pd.read_csv(notes_csv, usecols=["CREATE_BY"],
                                 chunksize=200_000, low_memory=False):
            for cb in chunk["CREATE_BY"].dropna().unique():
                tokens.update(t.lower() for t in
                              WORD.findall(str(cb).split(" - ", 1)[-1]))
    print(f"  {len(tokens):,} name tokens in the cohort + provider roster", file=sys.stderr)
    return tokens


def corpus_frequent_tokens(notes_csv: pathlib.Path, sample_rows: int,
                           min_frac: float, min_patients: int,
                           chunksize: int = 20_000) -> set:
    """Tokens that appear in the charts of a large fraction of distinct patients.

    A patient identifier is by construction low document-frequency: it appears
    in one patient's chart and nowhere else. So any token seen across a large
    share of patients is corpus vocabulary -- drug brand names, device names,
    template boilerplate -- and safe to allowlist.

    The notes file is grouped by patient, so a contiguous head of the file would
    cover only a sliver of the cohort. Rows are taken from every chunk instead,
    spreading the sample across the whole corpus, and the cut-off is a fraction
    of the patients actually seen rather than an absolute count.
    """
    if not notes_csv.exists():
        print(f"  skip corpus DF pass ({notes_csv} not found)", file=sys.stderr)
        return set()

    total_rows = sum(1 for _ in open(notes_csv, "rb")) - 1
    n_chunks = max(1, -(-total_rows // chunksize))
    per_chunk = max(1, sample_rows // n_chunks)

    df = collections.defaultdict(set)
    patients, rows = set(), 0
    for chunk in pd.read_csv(notes_csv, chunksize=chunksize,
                             usecols=["IP_PATIENT_ID", "NOTE_TEXT"],
                             low_memory=False):
        head = chunk.head(per_chunk)
        for pid, txt in zip(head["IP_PATIENT_ID"], head["NOTE_TEXT"].fillna("")):
            rows += 1
            patients.add(pid)
            for tok in set(WORD.findall(str(txt).lower())):
                if len(tok) >= 3:
                    df[tok].add(pid)
        print(f"\r  corpus DF: {rows:,} rows, {len(patients):,} patients, "
              f"{len(df):,} tokens", end="", file=sys.stderr)
    print(file=sys.stderr)

    cutoff = max(min_patients, int(min_frac * len(patients)))
    print(f"  cut-off: token must span >={cutoff} of {len(patients):,} patients",
          file=sys.stderr)
    return {t for t, pids in df.items() if len(pids) >= cutoff}


def english_tokens() -> set:
    """Lowercase dictionary words only.

    nltk's word corpus mixes in proper nouns (Ahmed, Chen, Abraham). Those are
    exactly the surnames the NER layer must stay free to redact, so entries that
    are not lowercase in the corpus are excluded.
    """
    try:
        from nltk.corpus import words as nltk_words
        return {w for w in nltk_words.words() if w.islower() and len(w) >= 3}
    except Exception as exc:
        print(f"  nltk words unavailable ({exc}); general tier will be empty", file=sys.stderr)
        return set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="Directory holding the study's structured CSVs")
    ap.add_argument("--out", default="deid2/resources")
    ap.add_argument("--min-count", type=int, default=1,
                    help="Minimum distinct catalog values a token must appear in")
    ap.add_argument("--notes-csv", default=None,
                    help="Provider notes CSV, for the corpus document-frequency pass")
    ap.add_argument("--ids-csv", default=None,
                    help="Patient_Identifiers.csv, so no real name is ever allowlisted")
    ap.add_argument("--sample-rows", type=int, default=150_000)
    ap.add_argument("--min-patient-frac", type=float, default=0.02,
                    help="Fraction of sampled patients a token must span to count "
                         "as corpus vocabulary")
    ap.add_argument("--min-patients", type=int, default=20,
                    help="Absolute floor for the above, for small samples")
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting catalog vocabulary...", file=sys.stderr)
    clinical = collect_catalog_tokens(data_dir, args.min_count)
    clinical |= {t.lower() for t in CURATED if len(t) >= 3}

    print("Collecting cohort name tokens...", file=sys.stderr)
    names = cohort_name_tokens(args.notes_csv,
                               args.ids_csv or data_dir / "Patient_Identifiers.csv")

    frequent = set()
    if args.notes_csv:
        print("Corpus document-frequency pass...", file=sys.stderr)
        frequent = corpus_frequent_tokens(pathlib.Path(args.notes_csv),
                                          args.sample_rows, args.min_patient_frac,
                                          args.min_patients)
        clinical |= frequent

    # Curated clinical eponyms are hand-vetted and kept even where they collide
    # with a real surname ("Bell" palsy, "Allen" test); the self-referential and
    # anchored-rule layers still redact those as names where they are names.
    curated = {t.lower() for t in CURATED if len(t) >= 3}
    dropped = (clinical - curated) & names
    clinical = (clinical - names) | curated
    print(f"  dropped {len(dropped):,} data-derived clinical tokens that are names",
          file=sys.stderr)

    # A staff surname that is also an everyday word ("Young", "Baker", "Long")
    # must not vanish from both tiers, or the NER layer would redact it in
    # running prose. Corpus frequency tells the two cases apart: keep such
    # tokens in the weak tier, where only confident detections override them.
    common_words = names & frequent
    # Seed the weak tier from corpus frequency as well as the dictionary. The
    # nltk word list omits most inflected forms -- "meal" is in it, "meals" is
    # not -- and "Meals" is also a surname on the provider roster, so without
    # this the roster layer redacts "TIMES A DAY WITH MEALS".
    general = (english_tokens() | frequent) - clinical - (names - common_words)
    print(f"  {len(common_words):,} names kept in the weak tier as common words",
          file=sys.stderr)

    (out_dir / "clinical_allowlist.txt").write_text("\n".join(sorted(clinical)) + "\n")
    (out_dir / "general_allowlist.txt").write_text("\n".join(sorted(general)) + "\n")
    print(f"\nclinical tier: {len(clinical):>7} tokens -> {out_dir/'clinical_allowlist.txt'}")
    print(f"general tier : {len(general):>7} tokens -> {out_dir/'general_allowlist.txt'}")


if __name__ == "__main__":
    main()

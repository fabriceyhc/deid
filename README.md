# Universal De-identification Script

## Introduction

This script provides a flexible command-line tool for de-identifying text data from various sources, including direct string input and columns within CSV files. It employs a multi-layered approach to Named Entity Recognition (NER) for identifying and masking Protected Health Information (PHI) and other sensitive data. Users can choose from Regex-based matching, SpaCy NER models, and Hugging Face Transformer-based NER models, or combine them for more comprehensive de-identification.

## Features

* **Multiple Masking Strategies:**
    * **Regex:** Utilizes a set of predefined regular expressions to identify common patterns (Dates, SSNs, Phone Numbers, Emails, MRNs, Addresses, and basic Person name patterns).
    * **SpaCy:** Leverages SpaCy's statistical NER models for broader entity detection.
    * **Hugging Face Transformers:** Employs state-of-the-art Transformer models for advanced NER tasks.
* **Flexible Input:** Process single text strings or entire columns in CSV files.
* **Configurable Maskers:** Choose which masker(s) to use (regex, spacy, huggingface) and specify models for SpaCy and Hugging Face.
* **Customizable Entity Targeting:** Specify which entity types to mask (e.g., PERSON, DATE, ORG).
* **Known Names List:** RegexMasker can use a JSON file of known first names, last names, and common male/female names to improve accuracy and reduce false positives for person names.
* **Customizable Mask:** Define the string used to replace identified entities (default: `[REDACTED]`).
* **Output Options:** Print de-identified text to console, save to a file, or create a new de-identified CSV file.
* **GPU Support:** Utilizes GPU for SpaCy and Hugging Face models if available and configured.

## Installation

### Prerequisites

* Python 3.7+

### Environment

#### 1. Install Miniconda (Recommended for Windows)

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage your Python environments, especially on Windows. Miniconda is a lightweight alternative to Anaconda and simplifies package and environment management.

- **Download Miniconda:**
  - Visit the [official Miniconda download page](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions) for quickstart instructions and alternative installers for all platforms.
  - Choose the installer appropriate for your operating system (Windows, macOS, or Linux).

- **Install Miniconda:**
  - Follow the instructions for your platform. On Windows, run:

```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o .\miniconda.exe
start /wait "" .\miniconda.exe /S
del .\miniconda.exe
```

#### 2. Create a Conda Environment with Python

After installing Miniconda, open your terminal (Anaconda Prompt on Windows, or your system terminal on macOS/Linux) and run:

```bash
conda create -n deid-env python=3.13
conda activate deid-env
```

This creates and activates a new environment named `deid-env` with Python 3.13. You can then proceed to install the required dependencies as described below.

### Dependencies

1.  **Clone the repository.**

2.  **Install required Python packages:**
    It's recommended to use a virtual environment.
    ```bash
    pip install pandas spacy transformers torch tqdm
    ```
    *(Note: `torch` is listed as it's a core dependency for Hugging Face Transformers. Depending on your system and if you intend to use a GPU, you might need a specific PyTorch build. Visit [pytorch.org](https://pytorch.org/) for installation instructions tailored to your OS and CUDA version if applicable.)*

3.  **Download SpaCy Models (if using SpaCy masker):**
    The script defaults to `en_core_web_trf`. You can download it or other models:
    ```bash
    python -m spacy download en_core_web_trf
    ```

## Usage

The script is run from the command line.

```bash
python deid.py [ARGUMENTS]
```


## Modes of Operation

You must specify either `--text` for single string input or `--input_csv` for CSV file processing.

* **Single Text Processing:**
    ```bash
    python deid.py --text "Your sensitive text here..." [OPTIONS]
    ```

* **CSV File Processing:**
    ```bash
    python deid.py --input_csv /path/to/input.csv --column_name "column_to_deid" --output_csv /path/to/output_deid.csv [OPTIONS]
    ```

## Command-Line Arguments

Below is a list of available command-line arguments (`[OPTIONS]`):

* `--text TEXT`
    * Description: A single text string to de-identify. (Mutually exclusive with `--input_csv`)
    * Default: `None`
* `--input_csv PATH`
    * Description: Path to the input CSV file for de-identification. (Mutually exclusive with `--text`)
    * Default: `None`
* `--column_name NAME`
    * Description: Name of the column to de-identify in the CSV. Required if `--input_csv` is used.
    * Default: `None`
* `--output_csv PATH`
    * Description: Path to save the de-identified CSV file. Required if `--input_csv` is used.
    * Default: `None`
* `--output_text_file PATH`
    * Description: Path to save the de-identified single text output. If not provided and `--text` is used, prints to console.
    * Default: `None`
* `--maskers M1 [M2 ...]`
    * Description: List of maskers to use. Choices: `regex`, `spacy`, `huggingface`.
    * Default: `regex`
* `--spacy_model NAME`
    * Description: Name of the SpaCy model to use (e.g., `en_core_web_sm`, `en_core_web_trf`).
    * Default: `en_core_web_trf`
* `--hf_models M1 [M2 ...]`
    * Description: Name(s) or path(s) of Hugging Face NER model(s).
    * Default: `StanfordAIMI/stanford-deidentifier-only-i2b2`
* `--hf_cache_dir PATH`
    * Description: Directory to cache Hugging Face models. Defaults to Transformers library default cache.
    * Default: `None`
* `--hf_device INT`
    * Description: Device for Hugging Face pipeline (e.g., `0` for GPU 'cuda:0', `-1` for CPU).
    * Default: `0` (attempts GPU)
* `--names_file PATH`
    * Description: Path to the JSON file containing known names for the RegexMasker.
    * Default: `./data/names.json`
* `--regex_debug`
    * Description: Enable debug printing for RegexMasker.
    * Default: `False` (this is a flag, presence enables it)
* `--entity_types T1 [T2 ...]`
    * Description: List of specific entity types to mask (e.g., `PERSON` `ORG` `DATE`). If not provided, maskers will target all entities they are configured for.
    * Default: `None`
* `--default_mask STR`
    * Description: The string to use for replacing identified PHI.
    * Default: `[REDACTED]`
* `-h`, `--help`
    * Description: Show help message and exit.

## Examples

Here are various examples demonstrating how to use `deid.py`:

### 1. De-identify a single string (regex default)
This is the simplest usage, relying on the default RegexMasker and printing to console.

```
python deid.py --text "Patient Johnathan Doe can be reached at john.doe@email.com or 123-456-7890. DOB is 01/02/1970."
```

### 2. De-identify a string using SpaCy and save to file
This example uses the SpaCy masker with the `en_core_web_sm` model, targets `PERSON` and `ORG` entities, and saves the output to `masked_output.txt`.

```
python deid.py --text "Call Dr. Smith at Mass General." --maskers spacy --spacy_model en_core_web_trf --output_text_file masked_output.txt --entity_types PERSON
```

### 3. Using all deidentification methods simultaneously
This example demonstrates a comprehensive approach using all available maskers (regex, spacy, huggingface)
```
python deid.py --input_csv patient_records.csv --column_name note_text --output_csv patient_records_deid.csv --maskers regex spacy huggingface
```

---

# Second Pass: `deid2`

`deid.py` above is the general-purpose, single-pass tool. `deid2` is a
second-pass pipeline built for corpora that have **already** been through a
de-identification step and still leak — the situation with the UCLA provider
notes, where the upstream pass replaced PHI with `___` but left a measurable
share of it behind.

## Why a second pass was needed

Measured on `Provider_Notes_deid.csv` (923,959 line rows, 1.55 GB of text),
against the `IP_PATIENT_ID -> PAT_NAME` crosswalk in `Patient_Identifiers.csv`:

| Residual PHI in the upstream output | Rate |
|---|---|
| Note lines containing **>=1 token of that patient's own name** | 8.4% |
| Note lines containing **>=2 tokens of that patient's own name** | 5.5% |
| Note lines with a street address | 2.6% |
| Note lines with a ZIP in state context | 6.1% |
| Note lines with a pager/extension number | 1.8% |
| `CREATE_BY` column | full provider name, in the clear, every row |

The failure is systematic rather than random. Template headers such as
`PATIENT: <name>  MRN: ___  DOB: ___` had the MRN and DOB removed and the name
left in place, and hyphenated surnames were half-redacted (`___-Thornbury, MD`).

## Design

Four layers, precision-descending. Each emits character spans independently;
`deid2.spans.resolve` arbitrates between them.

```
+--------------------------------------------------------------------+
| 1. SELF-REFERENTIAL GAZETTEER            precision ~1.0, ~0 compute |
|    The study already holds a patient -> name/DOB crosswalk, so a    |
|    patient identifier is looked up, not inferred. Matches nicknames |
|    (Robert -> Bob/Rob/Bert), initials, possessives, hyphen halves,  |
|    and DOB in eight written formats. Never vetoed.                  |
+--------------------------------------------------------------------+
| 2. ROSTER GAZETTEER                        Aho-Corasick, ~0 compute |
|    The full staff roster (3,988 surnames, 3,599 given names) is     |
|    harvested from the CREATE_BY column of the notes themselves --   |
|    no external name list. Requires capitalisation; short all-caps   |
|    hits are treated as clinical acronyms (POA, ROS), not surnames.  |
+--------------------------------------------------------------------+
| 3. CONTEXT RULES                                          regex, CPU|
|    Label-anchored patterns aimed at the failure modes measured      |
|    above, plus formats (phone, email, SSN, address, ZIP, month-name |
|    dates, pager/extension, ages 90+). Anchored hits are exempt from |
|    the allowlist veto; bare one-word hits are not.                  |
+--------------------------------------------------------------------+
| 4. TRANSFORMER NER                                    ~1.2 GPU-hr   |
|    obi/deid_roberta_i2b2 over overlapping 512-token windows. This   |
|    is the recall layer; precision is restored by the allowlist.     |
+--------------------------------------------------------------------+
```

### Arbitration

A span survives if it is plausible for its category, is not already a mask, and
is not vetoed by the allowlist. Two allowlist tiers do the veto:

* **clinical** (11.7k tokens) — drug names, lab components, ICD descriptions,
  eponyms, note-template vocabulary. Vetoes any vetoable span. This is what
  keeps `Kaposi sarcoma`, `vancomycin`, `fentanyl` and `methamphetamine` intact
  even when a model tags them with 0.99 confidence.
* **general** (202k tokens) — lowercase dictionary words. Vetoes only weak
  evidence (score < 0.90), so `Dr. Baker` is redacted while the word `baker` in
  running prose is not.

Both tiers are **derived from the study's own data**, not hand-written:
controlled vocabularies from the structured tables, plus a corpus
document-frequency pass. A token appearing across >=2% of patients cannot be a
patient identifier, so it is corpus vocabulary and safe to allowlist. Every
token belonging to a cohort patient or to the provider roster is subtracted
first, except a hand-vetted eponym list, so the allowlist can never shield a
real name.

### Output

Redactions are written as typed tags — `[NAME]`, `[DATE]`, `[AGE]`, `[PHONE]`,
`[EMAIL]`, `[ID]`, `[ADDRESS]`, `[CITY]`, `[ZIP]`, `[LOC]`, `[URL]` — which
remain distinguishable from the upstream `___`, so a later audit can tell which
pass caught what. `CREATE_BY` is replaced by a stable `PROV_<hash>` token that
still works as a grouping key.

Two defaults are deliberate and configurable:

* **Ages** are redacted only at 90 and above (`min_redacted_age`). Safe Harbor
  requires nothing more, and redacting `48 y.o.` would destroy clinically
  load-bearing text for no privacy gain.
* **Bare four-digit years** are kept (`redact_bare_years=False`). Safe Harbor
  permits years, and `Dx of AIDS in 2019` is worth keeping.

## Usage

```bash
# 1. Build the allowlists and the provider roster (once per data refresh)
python3 tools/build_allowlist.py \
    --data-dir  /path/to/original \
    --notes-csv /path/to/original/Provider_Notes_deid.csv \
    --out deid2/resources

# 2. Run the corpus job, sharded across GPUs by patient id
python3 run_deid_notes.py \
    --input-csv       /path/to/original/Provider_Notes_deid.csv \
    --identifiers-csv /path/to/original/Patient_Identifiers.csv \
    --roster          deid2/resources/provider_roster.txt \
    --out-dir         /path/to/processed/deid2 \
    --gpus 3,4,6

# 3. Measure what is left
python3 audit_deid.py \
    --notes-csv       /path/to/processed/deid2/Provider_Notes_deid2.csv \
    --identifiers-csv /path/to/original/Patient_Identifiers.csv \
    --note-ids-from   /path/to/processed/deid2/Provider_Notes_deid2.csv
```

Notes are reconstructed from their `LINE_NUMBER` rows before de-identification,
because PHI spans straddle those boundaries and the NER layer needs context.
Spans are then mapped back, so both `Provider_Notes_deid2.csv` (line-level,
original schema) and `Provider_Notes_full_deid2.csv` (note-level) are written.

Useful flags: `--no-ner` (rules and gazetteers only, CPU, ~50x faster),
`--limit N` (smoke test), `--keep-dates` (limited data set),
`--redact-hospitals`.

## Evaluation

Three tracks, none of which requires hand annotation.

### 1. Crosswalk recall -- real ground truth (`audit_deid.py`)

Because the patient -> name crosswalk is known, the share of notes still
containing their own patient's name is measured directly against ground truth.
Measured on 16,462 rows / 289 patients (1,526 positive cases):

| | Before | After |
|---|---|---|
| Notes with >=1 token of the patient's own name | 9.27% | **0.000%** |
| Notes with >=2 tokens of the patient's own name | 6.30% | **0.000%** |
| Patients with any leak | 162 / 289 | **0 / 289** |
| Street address / ZIP / MRN / age>89 | 2.3% / 6.4% / 0.3% / 0.12% | **0** |

### 2. Injection benchmark -- recall at scale (`eval_injection.py`)

Known PHI is injected into real notes at known offsets, using templates drawn
from the leak contexts observed in this corpus, then scored span-exactly.
5,000 notes x 10 injections = **97,380 labelled gold spans**:

| Category | Gold | Span-exact recall | Escaped |
|---|---:|---:|---:|
| DATE | 17,442 | 100.00% | 0 |
| PHONE | 12,606 | 100.00% | 0 |
| ADDRESS / ZIP / ID / AGE | 4,999 / 4,999 / 4,949 / 4,889 | 100.00% | 0 |
| EMAIL | 2,528 | 100.00% | 0 |
| NAME | 44,968 | 99.75% | 69 |
| **All** | **97,380** | **99.88%** | **69** |

Coverage is scored over the union of predicted spans and only over alphanumeric
characters, so a name legitimately redacted as two spans ("Vancamp, Arvin")
counts as covered.

### 3. Clinical-term preservation -- the precision proxy

Injected text cannot measure precision honestly: the substrate notes still
contain residual real PHI, so a redaction outside a gold span may well be
correct. Instead, terms that must never be redacted are counted before and
after on real corpus output. Over 29,760 notes / 122,559 term occurrences:

**Zero loss** for fentanyl, norfentanyl, methamphetamine, amphetamine,
methadone, buprenorphine, suboxone, naloxone, heroin, cocaine, cannabis,
benzodiazepine, opioid, and every syndrome and antibiotic term tracked.

The only apparent losses were `marijuana` (-37) and `alcohol` (-24), and both
are accounted for by the term appearing inside a redacted URL path
(`uclahealth.org/dementia/marijuana-use`): 41 and 26 such occurrences exist
respectively. Clinical narrative preservation is effectively 100%.

### Known residual

The 69 escaped names are cases with no nearby anchor that the NER layer also
scored below threshold, plus names the clinical allowlist deliberately wins on
-- `Glasgow` (Glasgow Coma Scale), `Bell`, `Allen`, `Park`. That trade is
intentional: the alternative is redacting "Glasgow Coma Scale" throughout the
corpus. The self-referential layer still catches those names for the patient
they belong to, which is the case that matters most.

Regression tests: `python3 -m pytest test/test_deid2.py -q` (40 tests).

## Library use

```python
from deid2 import build

deid = build(
    identifiers_csv="Patient_Identifiers.csv",
    roster_path="deid2/resources/provider_roster.txt",
    cache_dir="/local1/fabricehc/huggingface/hub",
    device="cuda:0",
)
clean = deid.deidentify(note_text, patient_id="IPPAT_...")
```

Pass `use_ner=False` for a CPU-only instance. `deid.plan_batch(texts, pids)`
returns the resolved spans instead of masked text, for auditing.

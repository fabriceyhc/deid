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

### 1. De-identify a single string 
This is the simplest usage, relying on the default RegexMasker and printing to console.

```
python deid.py --text "Patient Johnathan Doe can be reached at john.doe@email.com or 123-456-7890. DOB is 01/02/1970."
```

### 2. De-identify a string using SpaCy and save to file
This example uses the SpaCy masker with the `en_core_web_sm` model, targets `PERSON` and `ORG` entities, and saves the output to `masked_output.txt`.

```
python deid.py --text "Call Dr. Smith at Mass General." --maskers spacy --spacy_model en_core_web_trf --output_text_file masked_output.txt --entity_types PERSON
```

### 3. 
This example demonstrates a comprehensive approach using all available maskers (regex, spacy, huggingface)
```
python deid.py --input_csv patient_records.csv --column_name note_text --output_csv patient_records_deid.csv --maskers regex spacy huggingface
```
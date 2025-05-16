import re
import os
import spacy
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import unicodedata
import json
import argparse

#######################################################################
#  Maskers
#######################################################################

class RegexMasker:
    """
    A RegEx-based masker class that:
      - Normalizes text (NFKC, replaces fancy dashes and non-breaking spaces).
      - Uses a conservative multi-word capitalized pattern for 'NAME'.
      - After matching, ensures *all* tokens in the matched phrase are present in 'known_names'.
        This prevents something like 'Recreational Marijuna' from being treated as a name
        unless both words appear in 'known_names'.
    """

    def __init__(self, known_names=None, debug=False):
        """
        :param known_names: An optional list or set of valid name tokens (e.g., 'John', 'Doe').
        :param debug:       If True, prints debug info.
        """
        self.debug = debug
        self.known_names = set(known_names or [])

        self.patterns = [
            (r"\b(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", "PERSON"),
            (r"(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])[-/]((?:19|20)\d{2}|\d{2})", "DATE"),
            (r"\d{3}-\d{2}-\d{4}", "SSN"),
            (r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?=[^0-9]|$)", "PHONE"),
            (r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}(?=[^a-zA-Z0-9._%+\-]|$)", "EMAIL"),
            (r"(?:MRN\s*\d{5,}|\d{7,})", "MRN"),
            (r"\d{1,5}\s+(?:[A-Za-z0-9]+\s?){1,6},\s?[A-Za-z]+(?:[ -][A-Za-z]+)*,\s?[A-Z]{2},\s?\d{5}", "ADDRESS"),
        ]

    def get_entity_spans(self, text, entity_types=None):
        """
        Return a list of (start, end, label) for each detected entity.

        :param text:         The input text to search for patterns.
        :param entity_types: List or single string specifying which labels to return.
                             If None, returns all patterns.
        :return: A list of (start_index, end_index, label).
        """
        if isinstance(entity_types, str):
            entity_types = [entity_types]

        found_spans = []
        normalized_text = unicodedata.normalize("NFKC", text) # Basic normalization

        for pattern, label in self.patterns:
            if entity_types and label not in entity_types:
                continue

            for match in re.finditer(pattern, normalized_text):
                start, end = match.start(), match.end()
                matched_str = normalized_text[start:end]

                if label == "PERSON" and self.known_names:
                    tokens = re.findall(r"[A-Za-z]+", matched_str)
                    if not any(token in self.known_names for token in tokens):
                        continue

                found_spans.append((start, end, label))
                if self.debug:
                    print(f"[DEBUG] RegexMasker Matched '{matched_str}' as {label} from pattern {pattern}")
        return found_spans


class HuggingfaceMasker:
    def __init__(self, model_name="StanfordAIMI/stanford-deidentifier-only-i2b2", cache_dir=None, device=0):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=cache_dir)
        # device maps to 0 for 'cuda:0', -1 for 'cpu' in Hugging Face pipeline
        effective_device = device if device >= 0 else -1 # pipeline expects -1 for CPU
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=effective_device, aggregation_strategy="simple")


    def get_entity_spans(self, text, entity_types=None):
        if not text.strip():
            return []

        if isinstance(entity_types, str):
            entity_types = [entity_types]

        ner_results = self.nlp(text)
        # ner_results will be a list of dicts like:
        # [{'entity_group': 'PER', 'score': 0.99, 'word': 'John Doe', 'start': 8, 'end': 16}]
        # when aggregation_strategy="simple" or "first" or "max"

        spans = []
        for entity in ner_results:
            label = entity['entity_group'] # Use 'entity_group' with aggregation_strategy

            if entity_types is not None:
                if not any(e_type.lower() in label.lower() for e_type in entity_types):
                    continue
            spans.append((entity["start"], entity["end"], label))
        return spans


class SpaCyNERMasker:
    def __init__(self, model_name="en_core_web_trf"):
        try:
            if spacy.prefer_gpu():
                 print(f"SpaCy is attempting to use GPU for model '{model_name}'.")
            else:
                 print(f"SpaCy GPU not available or not preferred for model '{model_name}', using CPU.")
        except Exception as e:
            print(f"SpaCy GPU preference check failed for model '{model_name}': {e}. Will proceed with CPU or default.")
        self.nlp = spacy.load(model_name)

    def get_entity_spans(self, text, entity_types=None):
        if not text.strip():
            return []

        if isinstance(entity_types, str):
            entity_types = [entity_types]

        doc = self.nlp(text)
        entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

        if entity_types is not None:
            filtered = []
            for start, end, label in entities:
                # Exact match for SpaCy labels often preferred, but keeping partial for consistency with original
                if any(e_type.lower() in label.lower() for e_type in entity_types):
                    filtered.append((start, end, label))
            entities = filtered
        return entities

#######################################################################
# Analysis Functions
#######################################################################

def load_names(file_path: str = "./data/names.json", as_set: bool = True):
    """
    Loads a local names.json file and returns a unified list (or set) of names.
    Adjusted to use "men", "women", and "last" keys from the provided JSON structure.

    :param file_path: The local path to 'names.json'.
    :param as_set:    If True, return the names as a set; otherwise, return them as a list.

    :return: A set or list of names combined from the 'men', 'women', and 'last' lists.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Names file not found at {file_path}. RegexMasker will operate without a known names list if used.")
        return set() if as_set else []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            names_dict = json.load(f)
        # *** ADJUSTED KEYS HERE ***
        combined = (
            names_dict.get("women", []) +  # Changed from "female"
            names_dict.get("men", []) +    # Changed from "male"
            names_dict.get("last", [])
        )
        return set(combined) if as_set else combined
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from names file {file_path}: {e}. RegexMasker will operate without a known names list.")
        return set() if as_set else []
    except Exception as e:
        print(f"An unexpected error occurred while loading names from {file_path}: {e}. RegexMasker will operate without a known names list.")
        return set() if as_set else []


def replace_consecutive_NER_tags(text, mask_tag):
    escaped_mask_tag = re.escape(mask_tag)
    pattern = r'(' + escaped_mask_tag + r')(?:\s*\1)+'
    replacement = r'\1'
    result_text = re.sub(pattern, replacement, text)
    return result_text

def merge_spans(spans):
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: x[0])
    merged = []
    if not spans:
        return []

    current_start, current_end, current_label = spans[0]

    for i in range(1, len(spans)):
        next_start, next_end, next_label = spans[i]
        if next_start < current_end:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end, current_label))
            current_start, current_end, current_label = next_start, next_end, next_label
    merged.append((current_start, current_end, current_label))
    return merged

def mask_text(text, spans, mask='[REDACTED]'):
    if not spans:
        return text
    result = []
    last_end = 0
    for start, end, label in spans:
        if start < last_end:
            print(f"Warning: Overlapping span detected in mask_text after merge. Skipping span: ({start}, {end}, '{label}')")
            continue
        result.append(text[last_end:start])
        result.append(mask)
        last_end = end
    result.append(text[last_end:])
    return "".join(result)


class Deidentifier:
    def __init__(self, maskers, default_mask='[REDACTED]'):
        self.maskers = maskers
        self.default_mask = default_mask
        tqdm.pandas(desc="De-identifying")

    def deidentify(self, text, entity_types=None):
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            return text

        all_spans = []
        for masker_instance in self.maskers:
            try:
                spans = masker_instance.get_entity_spans(text, entity_types=entity_types)
                all_spans.extend(spans)
            except Exception as e:
                print(f"Error during get_entity_spans for {type(masker_instance).__name__}: {e}")


        filtered_spans = []
        for start, end, label in all_spans:
            token_text = text[start:end]
            if token_text.isupper() and ('person' not in label.lower()):
                continue
            filtered_spans.append((start, end, label))

        merged_spans = merge_spans(filtered_spans)
        masked_text = mask_text(text, merged_spans, mask=self.default_mask)
        masked_text = replace_consecutive_NER_tags(masked_text, self.default_mask)
        return masked_text

    def deidentify_csv(self, input_csv_path, output_csv_path, column_name, entity_types=None):
        try:
            df = pd.read_csv(input_csv_path)
        except FileNotFoundError:
            print(f"Error: Input CSV file not found at {input_csv_path}")
            return
        except Exception as e:
            print(f"Error reading CSV file {input_csv_path}: {e}")
            return

        if column_name not in df.columns:
            print(f"Error: Column '{column_name}' not found in CSV. Available columns: {df.columns.tolist()}")
            return

        df[column_name] = df[column_name].astype(str).fillna('')

        df[column_name] = df[column_name].progress_apply(
            lambda x: self.deidentify(x, entity_types=entity_types)
        )

        try:
            df.to_csv(output_csv_path, index=False)
            print(f"De-identified data successfully saved to {output_csv_path}")
        except Exception as e:
            print(f"Error writing de-identified CSV to {output_csv_path}: {e}")

#######################################################################
# Main Entry Point
#######################################################################

def main():
    parser = argparse.ArgumentParser(description="De-identify text in strings or CSV files using various masking techniques.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="A single text string to de-identify.")
    group.add_argument("--input_csv", type=str, help="Path to the input CSV file for de-identification.")

    parser.add_argument("--column_name", type=str, help="Name of the column to de-identify in the CSV. Required if --input_csv is used.")
    parser.add_argument("--output_csv", type=str, help="Path to save the de-identified CSV file. Required if --input_csv is used.")
    parser.add_argument("--output_text_file", type=str, help="Path to save the de-identified single text output. If not provided and --text is used, prints to console.")

    parser.add_argument("--maskers", nargs='+', choices=['regex', 'spacy', 'huggingface'],
                        default=['regex', 'spacy', 'huggingface'], help="List of maskers to use (e.g., regex spacy huggingface). Default: ['regex']")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf",
                        help="Name of the SpaCy model to use (e.g., en_core_web_sm, en_core_web_trf). Default: en_core_web_trf")
    parser.add_argument("--hf_models", nargs='+', default=["StanfordAIMI/stanford-deidentifier-only-i2b2"],
                        help="Name(s) or path(s) of Hugging Face NER model(s). Default: ['StanfordAIMI/stanford-deidentifier-only-i2b2']")
    parser.add_argument("--hf_cache_dir", type=str, default=None,
                        help="Directory to cache Hugging Face models. Defaults to Transformers library default cache.")
    parser.add_argument("--hf_device", type=int, default=0,
                        help="Device for Hugging Face pipeline (e.g., 0 for GPU 'cuda:0', -1 for CPU). Default: 0 (attempts GPU)")
    parser.add_argument("--names_file", type=str, default="./data/names.json", # Default path
                        help="Path to the JSON file containing known names for the RegexMasker. Default: ./data/names.json")
    parser.add_argument("--regex_debug", action='store_true', help="Enable debug printing for RegexMasker.")

    parser.add_argument("--entity_types", nargs='*', default=None,
                        help="List of specific entity types to mask (e.g., PERSON ORG DATE). If not provided, maskers will target all entities they are configured for or find.")
    parser.add_argument("--default_mask", type=str, default="[REDACTED]",
                        help="The string to use for replacing identified PHI. Default: '[REDACTED]'")

    args = parser.parse_args()

    if args.input_csv and not (args.output_csv and args.column_name):
        parser.error("--input_csv requires --output_csv and --column_name to be specified.")

    active_maskers = []
    print("Initializing maskers...")
    if "regex" in args.maskers:
        # The load_names function is called here with the path from args
        known_names_list = load_names(args.names_file)
        active_maskers.append(RegexMasker(known_names=known_names_list, debug=args.regex_debug))
        print(f"RegexMasker initialized {'with debug.' if args.regex_debug else '.'} Using names file: {args.names_file}")
    if "spacy" in args.maskers:
        try:
            active_maskers.append(SpaCyNERMasker(model_name=args.spacy_model))
            print(f"SpaCyNERMasker initialized with model: {args.spacy_model}.")
        except OSError as e:
            print(f"Critical Error: Failed to initialize SpaCy model '{args.spacy_model}': {e}")
            print("Please ensure the SpaCy model is downloaded (e.g., 'python -m spacy download en_core_web_trf'). Exiting.")
            return
        except Exception as e:
            print(f"Critical Error: An unexpected error occurred while initializing SpaCy model '{args.spacy_model}': {e}. Exiting.")
            return
    if "huggingface" in args.maskers:
        for model_name in args.hf_models:
            try:
                active_maskers.append(HuggingfaceMasker(model_name=model_name, cache_dir=args.hf_cache_dir, device=args.hf_device))
                print(f"HuggingfaceMasker initialized with model: {model_name} (cache: {args.hf_cache_dir}, device: {args.hf_device}).")
            except Exception as e:
                print(f"Critical Error: Failed to initialize Hugging Face model '{model_name}': {e}. Exiting.")
                return

    if not active_maskers:
        print("No maskers were specified or successfully initialized. Nothing to do. Exiting.")
        return

    deidentifier = Deidentifier(active_maskers, default_mask=args.default_mask)
    print(f"Deidentifier initialized with {len(active_maskers)} masker(s) and mask string '{args.default_mask}'.")
    
    entity_types_to_process = args.entity_types
    if entity_types_to_process:
        print(f"Targeting specific entity types: {', '.join(entity_types_to_process)}")
    else:
        print("No specific entity types provided; maskers will use their default detection scope.")


    if args.input_csv:
        print(f"\nProcessing CSV file: {args.input_csv}")
        print(f"De-identifying column: '{args.column_name}'")
        print(f"Output will be saved to: {args.output_csv}")
        deidentifier.deidentify_csv(
            input_csv_path=args.input_csv,
            output_csv_path=args.output_csv,
            column_name=args.column_name,
            entity_types=entity_types_to_process
        )
    elif args.text:
        print(f"\nOriginal text: \"{args.text}\"")
        masked_text = deidentifier.deidentify(args.text, entity_types=entity_types_to_process)
        print(f"Masked text:   \"{masked_text}\"")
        if args.output_text_file:
            try:
                with open(args.output_text_file, 'w', encoding='utf-8') as f:
                    f.write(masked_text)
                print(f"Masked text saved to {args.output_text_file}")
            except IOError as e:
                print(f"Error: Could not write masked text to file {args.output_text_file}: {e}")

    print("\nDe-identification process complete.")

if __name__ == "__main__":
    # Example Usages:
    # 1. De-identify a single string and print to console (using default regex masker):
    #    python deid.py --text "Patient Johnathan Doe can be reached at john.doe@email.com or 123-456-7890. DOB is 01/02/1970."
    #
    # 2. De-identify a string using SpaCy and save to file:
    #    python deid.py --text "Call Dr. Smith at Mass General." --maskers spacy --spacy_model en_core_web_sm --output_text_file masked_output.txt --entity_types PERSON ORG
    #
    # 3. De-identify a CSV column using regex and a Hugging Face model:
    #    python deid.py --input_csv notes.csv --column_name clinical_note --output_csv notes_deid.csv --maskers regex huggingface --hf_models "Jean-Baptiste/roberta-large-ner-english" --names_file custom_names.json
    #
    # 4. De-identify a CSV using all maskers and specific entity types:
    #    python deid.py --input_csv patient_records.csv --column_name summary --output_csv patient_records_deid.csv --maskers regex spacy huggingface --entity_types PERSON DATE PHONE EMAIL ADDRESS ID MRN --default_mask "[PHI]"
    
    main()
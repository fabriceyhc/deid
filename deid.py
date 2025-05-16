import re
import os
import spacy
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import unicodedata
import json

#######################################################################
#  Maskers                                                            
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

        # This pattern matches 2-3 capitalized "words":
        #   1) \b(?P<name>[A-Z][a-z]+   -> first capitalized word
        #   2) (?:\s+[A-Z][a-z]+)      -> second capitalized word
        #   3) (?:\s+[A-Z][a-z]+)?     -> optional third capitalized word
        #
        # You can adapt it to allow single-letter initials with periods if desired,
        # or refine it further for hyphenated names, etc.
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

        # Convert entity_types to list if it's just a string
        if isinstance(entity_types, str):
            entity_types = [entity_types]

        found_spans = []
        for pattern, label in self.patterns:
            # Skip if user requested specific labels
            if entity_types and label not in entity_types:
                continue

            for match in re.finditer(pattern, text):
                start, end = match.start(), match.end()
                matched_str = text[start:end]

                # If this is a NAME match, enforce the known-names check
                if label == "PERSON" and self.known_names:
                    # Extract alphabetical tokens
                    tokens = re.findall(r"[A-Za-z]+", matched_str)
                    # If NONE of the tokens is in known_names, skip
                    if not any(token in self.known_names for token in tokens):
                        continue

                found_spans.append((start, end, label))
                if self.debug:
                    print(f"[DEBUG] Matched '{matched_str}' as {label}")

        return found_spans
    

class HuggingfaceMasker:
    def __init__(self, model_name="StanfordAIMI/stanford-deidentifier-only-i2b2", cache_dir="./models"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) #, cache_dir=cache_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=0)

    def get_entity_spans(self, text, entity_types=None):
        """
        Return a list of (start, end, label) for each detected entity.
        If entity_types is provided, only return spans that match
        (case-insensitive, partial substring) any of them.
        """
        if not text.strip():
            return []

        # Normalize entity_types to a list
        if isinstance(entity_types, str):
            entity_types = [entity_types]

        ner_results = self.nlp(text)
        ner_results = sorted(ner_results, key=lambda x: x['start'])

        spans = []
        for entity in ner_results:
            label = entity['entity']
            
            # If entity_types is specified, check partial substring match, ignoring case
            if entity_types is not None:
                if not any(e_type.lower() in label.lower() for e_type in entity_types):
                    continue

            spans.append((entity["start"], entity["end"], label))
        return spans
    

class SpaCyNERMasker:
    def __init__(self, model_name="en_core_web_trf"):
        spacy.prefer_gpu()
        self.nlp = spacy.load(model_name)

    def get_entity_spans(self, text, entity_types=None):
        """
        Return a list of (start, end, label) for each detected entity.
        If entity_types is provided, only return spans matching any of them
        (case-insensitive, partial substring). Also removes 'PRODUCT' entities.
        """
        if not text.strip():
            return []

        # Normalize entity_types to a list
        if isinstance(entity_types, str):
            entity_types = [entity_types]

        doc = self.nlp(text)
        entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

        # Filter by partial substring match, ignoring case
        if entity_types is not None:
            filtered = []
            for start, end, label in entities:
                if any(e_type.lower() in label.lower() for e_type in entity_types):
                    filtered.append((start, end, label))
            entities = filtered

        return entities
    
#######################################################################
# Analysis Functions
#######################################################################
    
    import json

def load_names(file_path: str = "./data/names.json", as_set: bool = True):
    """
    Loads a local names.json file (with 'female', 'male', and 'last' lists)
    and returns a unified list (or set) of names.

    :param file_path: The local path to 'names.json'.
    :param as_set:    If True, return the names as a set; otherwise, return them as a list.

    :return: A set or list of names combined from the 'female', 'male', and 'last' lists.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        names_dict = json.load(f)

    combined = (
        names_dict.get("female", []) +
        names_dict.get("male", []) +
        names_dict.get("last", [])
    )
    
    return set(combined) if as_set else combined

def replace_consecutive_NER_tags(text):
    # Regex pattern to match substrings in brackets and find consecutive occurrences
    pattern = r'(\[\w+\])(\1)+'
    replacement = r'\1'
    result_text = re.sub(pattern, replacement, text)
    return result_text

def merge_spans(spans):
    """
    Merge overlapping or contiguous spans into a single span.
    """
    if not spans:
        return []
    # Sort by start
    spans = sorted(spans, key=lambda x: x[0])
    merged = []
    cur_start, cur_end, cur_label = spans[0]

    for i in range(1, len(spans)):
        s, e, l = spans[i]
        if s <= cur_end:  # overlapping or contiguous
            cur_end = max(cur_end, e)
            # Optionally, could combine labels
        else:
            merged.append((cur_start, cur_end, cur_label))
            cur_start, cur_end, cur_label = s, e, l
    merged.append((cur_start, cur_end, cur_label))
    return merged

def mask_text(text, spans, mask='[REDACTED]'):
    """
    Given a text and a list of spans (start,end,label), mask those spans out.
    """
    if not spans:
        return text
    result = []
    last_end = 0
    for start, end, label in spans:
        result.append(text[last_end:start])
        result.append(mask)
        last_end = end
    result.append(text[last_end:])
    return "".join(result)


class Deidentifier:
    def __init__(self, maskers, default_mask='___'):
        """
        :param maskers: a list of instantiated masker objects (HuggingfaceMasker, SpaCyNERMasker, etc.)
        :param default_mask: the default mask string to use for PHI
        """
        self.maskers = maskers
        self.default_mask = default_mask
        tqdm.pandas()

    def deidentify(self, text, entity_types=None):
        """
        De-identify text by masking spans identified by all maskers.
        If entity_types is specified, only mask those entity types
        (using case-insensitive, partial substring match).
        """
        if not text.strip():
            return text

        all_spans = []
        for m in self.maskers:
            spans = m.get_entity_spans(text, entity_types=entity_types)
            all_spans.extend(spans)

        # Preserve acronyms (all caps) if NOT labeled as person (case-insensitive, partial substring match)
        filtered_spans = []
        for start, end, label in all_spans:
            token_text = text[start:end]
            # If token_text is ALL CAPS and the label does not contain 'PERSON' (ignore case),
            # skip masking it.
            if token_text.isupper() and ('person' not in label.lower()):
                continue
            filtered_spans.append((start, end, label))

        merged_spans = merge_spans(filtered_spans)
        masked_text = mask_text(text, merged_spans, mask=self.default_mask)
        masked_text = replace_consecutive_NER_tags(masked_text)
        return masked_text

    def deidentify_csv(self, input_csv_path, output_csv_path, column_name, entity_types=None):
        """
        1. Reads a CSV file from input_csv_path
        2. Applies de-identification to 'column_name'
        3. Saves the de-identified data to output_csv_path
        
        :param input_csv_path: Path to the input CSV file
        :param output_csv_path: Path where the output CSV file will be saved
        :param column_name: The name of the column in which we have the text to de-identify
        :param entity_types: list/string of entity types to mask. 
                            (e.g. ["PERSON", "ORG"]) or just "PERSON"
        """
        # Read the CSV file
        df = pd.read_csv(input_csv_path)

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in CSV file.")

        # Apply de-identification row by row
        def deid_func(text):
            # Handle missing or NaN values gracefully
            if not isinstance(text, str):
                return text
            return self.deidentify(text, entity_types=entity_types)

        df[column_name] = df[column_name].progress_apply(deid_func)

        # Save the updated DataFrame to a new CSV
        df.to_csv(output_csv_path, index=False)

#######################################################################
# Main Entry Point
#######################################################################

if __name__ == "__main__":

    # CUDA_VISIBLE_DEVICES=0 python -m deid

    # spacy_masker = SpaCyNERMasker(model_name="en_core_web_sm") # "en_core_web_trf" "en_core_web_sm"
    # hf_masker_1  = HuggingfaceMasker(model_name=".\models\stanford-deidentifier-base")
    # hf_masker_2  = HuggingfaceMasker(model_name=".\models\deid_roberta_i2b2")
    regex_masker = RegexMasker(known_names=load_names(), debug=False)

    # Instantiate the Deidentifier class with a list of maskers
    deidentifier = Deidentifier([regex_masker])
    # deidentifier = Deidentifier([spacy_masker, regex_masker])
    # deidentifier = Deidentifier([spacy_masker, hf_masker_1, hf_masker_2])

    # examples = [
    #     "Patient denies snorting Klonopin, but Fabrice is skeptical.",
    #     "Patient denies snorting Xanax, but Panayiotis is skeptical.",
    #     "Patient John Doe (DOB: 01/01/1980) was admitted with symptoms of severe anxiety and was prescribed Xanax for immediate relief.",
    #     "On 03/15/2024, Jane Smith was seen by Dr. Emily Johnson at Lakeside Medical Center for a follow-up regarding her chronic insomnia. Ambien dosage was adjusted.",
    #     "Patient Mr. Brown, a 45-year-old with a history of type 2 diabetes, was advised to continue with their current medication, Metformin, and to monitor their blood sugar levels closely.",
    #     "The blood test results from 04/10/2024 for Alex Martinez indicate a need to modify the current treatment plan to include Lipitor for better cholesterol management.",
    #     "Patient Ms. Wilson's insurance provider, HealthFirst Insurance, has approved coverage for the prescribed Humira starting from 05/01/2024.",
    #     "During the consultation on 02/20/2024, Lisa Chang disclosed a family history of glaucoma. Considering this, Timolol eye drops were prescribed as a precautionary measure.",
    #     "Patient Lee underwent a successful laparoscopic cholecystectomy on 01/25/2024 and is scheduled for a follow-up on 02/15/2024 to assess the need for further medication adjustments, including the possibility of incorporating Ursodiol.",
    # ]

    # for i, ex in enumerate(examples, 1):
    #     masked = deidentifier.deidentify(ex)
    #     print(f"Example {i}:\nOriginal: {ex}\nMasked:   {masked}\n")

    base_path = "F:/Inbound/CTSI/CLShover_24_22-001273/Data"
    
    PHI_candidates = [
        ("Problem_Lists.csv", "PROBLEM_DESCRIPTION"),       # finished on full pipeline with en_core_web_trf + regex
        ("Social_History.csv", "ILLICIT_DRUG_COMMENTS"),    # finished on full pipeline with en_core_web_trf + regex
        ("Social_History.csv", "ALCOHOL_COMMENTS"),         # finished with regex only
        ("Provider_Notes.csv", "NOTE_TEXT"),                # finished with regex only
        ("Labs.csv", "RESULT"),                             # finished on full pipeline with en_core_web_trf + regex
    ]

    hipaa_ner_labels = [
        "PERSON",       # For names (patient, provider, etc.)
        "GPE",          # For city/state/region, etc.
        "LOCATION",     # Other location labels
        "ORG",          # Organizations (e.g., hospitals, clinics, employers)
        "DATE",         # Dates (DOB, admission date, etc.)
        "TIME",         # Times, if your model distinguishes them separately
        "PHONE",        # Phone numbers
        "EMAIL",        # Emails
        "CONTACT",      # Could include phone, fax, email, etc.
        "ID",           # Could include SSN, MRN, account, license, etc.
        "SSN",          # Social Security Number specifically
        "MRN",          # Medical Record Number specifically
        "URL",          # Websites or IP addresses
        "FAC",          # Facility names (some models use FAC for “facility”)
        "PRODUCT",      # Sometimes used for device identifiers
        "LICENSE",      # Certificate/license numbers (if your model uses this label)
    ]

    for filename, column in PHI_candidates:
        deidentifier.deidentify_csv(
            input_csv_path=os.path.join(base_path, filename),
            output_csv_path=os.path.join(base_path, filename.replace(".csv", "_re.csv")),
            column_name=column,
            entity_types=hipaa_ner_labels
        )

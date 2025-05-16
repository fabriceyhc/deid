#!/usr/bin/env python3
"""
analyze_maskers.py
------------------
A standalone script to analyze overlap and frequency of detected spans
across multiple NER or PHI maskers (e.g., SpaCy, HuggingFace) using
text pulled from a CSV file.

Usage:
  python analyze_maskers.py --csv-file=F:/Inbound/CTSI/CLShover_24_22-001273/Data/Problem_Lists.csv --column=PROBLEM_DESCRIPTION --masker=spacy --masker=hf_stanford --masker=hf_obi --masker=regex
  python analyze_maskers.py --csv-file=F:/Inbound/CTSI/CLShover_24_22-001273/Data/Problem_Lists.csv --column=PROBLEM_DESCRIPTION --masker=spacy --masker=regex
"""

import argparse
import os
from collections import defaultdict
import pandas as pd
import re

# -------------------------------------------------------------------------
# If your maskers are defined in another module, replace these lines with
# the proper imports, e.g.:
#   from my_deid_code import SpaCyNERMasker, HuggingfaceMasker
# -------------------------------------------------------------------------
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

#######################################################################
#  Maskers                                                            #
#######################################################################
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
    
class RegexMasker:
    """
    A RegEx-based masker class that tries to detect many common HIPAA-protected
    data elements (PHI): names, phone numbers, emails, SSNs, MRNs, addresses, etc.
    It returns entity spans (start, end, label), which can then be masked out
    by a deidentifier pipeline.
    """

    def __init__(self):
        # Each tuple is (regex_pattern, entity_label).
        # Adjust/add patterns as needed for your data.
        self.patterns = [
            # (1) NAME pattern with negative lookbehind to skip first word of a sentence.
            #     Explanation:
            #       - (?<!^)       : negative lookbehind to avoid matching if it's at the very start of the entire text
            #       - (?<![.!?]\s) : negative lookbehind to avoid matching if it's right after a period, exclamation, or question mark + space
            #       - \b[A-Z][a-z]+(\s[A-Z][a-z]+)* : simple pattern for capitalized words (or multiple words)
            #     Caveat: This will also skip *true* names if they happen to be the first word of a sentence.
            (r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", "NAME"),

            # 2) Dates (MM/DD/YYYY or MM-DD-YYYY)
            #    This will catch many date formats in the US style.
            #    For other date formats (e.g., DD/MM/YYYY), adjust accordingly.
            (r"\b(0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])[/-](19|20)\d{2}\b", "DATE"),

            # 3) Social Security Number (SSN) - US style
            (r"\b\d{3}-\d{2}-\d{4}\b", "ID"),

            # 4) Phone Numbers (US style)
            #    This should catch patterns like:
            #    123-456-7890, (123) 456-7890, 123.456.7890, etc.
            (r"\b(\(\d{3}\)|\d{3})([\s\-.])?\d{3}([\s\-.])?\d{4}\b", "PHONE"),

            # 5) Emails
            (r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b", "EMAIL"),

            # 6) Medical Record Numbers (MRN)
            #    This is a placeholder pattern that can catch either:
            #      - 'MRN' followed by a number, or
            #      - A 7+ digit sequence (some institutions use 7 or more digits).
            #    Adjust as appropriate for your environment.
            (r"\b(?:MRN\s*\d{5,}|\d{7,})\b", "ID"),

            # 7) US Addresses (highly approximate)
            #    Matches a street number (up to 5 digits) + street name words,
            #    then a comma + city + comma + 2-letter state + comma + 5-digit ZIP.
            #    Example: "12345 Example Road, Springfield, IL, 62704"
            (r"\b\d{1,5}\s+(?:[A-Za-z0-9]+\s?){1,6},\s?[A-Za-z]+(?:[ -][A-Za-z]+)*,\s?[A-Z]{2},\s?\d{5}\b", "ADDRESS"),
        ]

    def get_entity_spans(self, text, entity_types=None):
        """
        Return a list of (start, end, label) for each detected entity.

        :param text:          The input text to search for patterns.
        :param entity_types:  A list (or single string) of entity labels to include.
                              If None, returns all detected spans.
                              Example: ["PHONE", "EMAIL"] or "ADDRESS"
        
        :return: A list of tuples: (start_index, end_index, label)
        """
        if not isinstance(text, str) or not text:
            return []

        # If the user passed a single string, convert it to a list for convenience
        if isinstance(entity_types, str):
            entity_types = [entity_types]

        found_spans = []
        for pattern, label in self.patterns:
            # If entity_types is specified, skip patterns that don't match
            if entity_types is not None and label not in entity_types:
                continue

            for match in re.finditer(pattern, text):
                start, end = match.start(), match.end()
                found_spans.append((start, end, label))

        return found_spans

#######################################################################
# Analysis Functions
#######################################################################
def analyze_maskers_across_texts(texts, maskers, masker_names=None, entity_types=None):
    """
    Analyze how each masker performs on a list of texts. We gather spans from
    each masker, compute overlap, and summarize statistics.

    :param texts: A list of strings to analyze.
    :param maskers: A list of instantiated maskers (SpaCyNERMasker, HuggingfaceMasker, etc.).
    :param masker_names: (Optional) a list of names (strings) to label each masker in the output.
                        If omitted, we'll just use "Masker_1", "Masker_2", etc.
    :param entity_types: (Optional) only consider spans of specific entity labels (e.g., ["PERSON"]).
    :return: (summary, results)
             summary => dict with aggregated stats
             results => list of per-text details
    """
    if masker_names is None or len(masker_names) != len(maskers):
        masker_names = [f"Masker_{i+1}" for i in range(len(maskers))]

    # For aggregated stats
    masker_counts = defaultdict(int)       # total spans found per masker
    overlap_counts = defaultdict(int)      # total overlap counts between pairs

    # Detailed results per text
    results = []

    for text_id, text in enumerate(texts):
        # Gather spans from each masker => store them as sets for easy intersection
        masker_spans = {}
        for i, masker in enumerate(maskers):
            name = masker_names[i]
            spans = masker.get_entity_spans(text, entity_types=entity_types)
            masker_spans[name] = set(spans)
            masker_counts[name] += len(spans)

        # Compare each pair of maskers
        pairwise_overlaps = {}
        for i in range(len(maskers)):
            for j in range(i+1, len(maskers)):
                name_i = masker_names[i]
                name_j = masker_names[j]
                intersect = masker_spans[name_i].intersection(masker_spans[name_j])
                pairwise_overlaps[f"{name_i} & {name_j}"] = len(intersect)
                overlap_counts[(name_i, name_j)] += len(intersect)

        # Store per-text detail
        text_result = {
            "text_id": text_id,
            "text": text,
            "masker_spans": {name: list(masker_spans[name]) for name in masker_spans},
            "pairwise_overlaps": pairwise_overlaps
        }
        results.append(text_result)

    summary = {
        "total_spans_per_masker": dict(masker_counts),
        "total_pairwise_overlaps": {
            f"{k[0]} & {k[1]}": v for k, v in overlap_counts.items()
        }
    }
    return summary, results

def decide_if_masker_is_redundant(summary, main_masker_name, other_masker_name):
    """
    Check if 'other_masker_name' is "redundant" because 'main_masker_name'
    always catches all of its spans. i.e. overlap == total spans of other masker
    """
    pairwise_key = f"{main_masker_name} & {other_masker_name}"
    reverse_key  = f"{other_masker_name} & {main_masker_name}"

    total_other_spans = summary["total_spans_per_masker"].get(other_masker_name, 0)
    # Overlap might be stored in either direction
    if pairwise_key in summary["total_pairwise_overlaps"]:
        overlap_val = summary["total_pairwise_overlaps"][pairwise_key]
    else:
        overlap_val = summary["total_pairwise_overlaps"].get(reverse_key, 0)

    return overlap_val == total_other_spans

#######################################################################
# Main Entry Point
#######################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Analyze NER / PHI Maskers overlap on a set of texts read from a CSV."
    )
    parser.add_argument("--csv-file", type=str, required=True,
                        help="Path to the CSV file containing your texts.")
    parser.add_argument("--column", type=str, required=True,
                        help="The column name in the CSV that contains the text to analyze.")
    parser.add_argument("--masker", type=str, action="append", required=True,
                        help="Which maskers to use. Possible values: 'spacy', 'hf_stanford', 'hf_obi', etc. "
                             "You can supply multiple --masker arguments.")

    args = parser.parse_args()

    # 1) Load the CSV
    if not os.path.isfile(args.csv_file):
        print(f"Error: file not found: {args.csv_file}")
        exit(1)

    df = pd.read_csv(args.csv_file).sample(n=10000)
    if args.column not in df.columns:
        print(f"Error: column '{args.column}' not found in CSV.")
        exit(1)

    # Drop rows where the column is NaN or empty
    df = df.dropna(subset=[args.column])
    texts = df[args.column].astype(str).tolist()
    texts = [
        "Admitted on 12 Jan 2023, patient MRN 1234567, phone 2023 123 123 at 123/45 Fake Street, City, State,2024"
    ]

    # 2) Instantiate the requested maskers & build display names
    maskers = []
    masker_names = []
    for m in args.masker:
        if m.lower() == "spacy":
            maskers.append(SpaCyNERMasker(model_name="en_core_web_trf"))
            masker_names.append("spaCy")
        elif m.lower() == "hf_stanford":
            maskers.append(HuggingfaceMasker(model_name=".\models\stanford-deidentifier-base"))
            masker_names.append("HF_Stanford")
        elif m.lower() == "hf_obi":
            maskers.append(HuggingfaceMasker(model_name=".\models\deid_roberta_i2b2"))
            masker_names.append("HF_obi")
        elif m.lower() == "regex":
            maskers.append(RegexMasker())
            masker_names.append("Regex")
        else:
            print(f"Warning: unknown masker option '{m}'. Skipping...")
            continue

    if not maskers:
        print("Error: no valid maskers specified.")
        exit(1)
    
    hipaa_ner_labels = [
        "PERSON",       # For names (patient, provider, etc.)
        "GPE",          # For city/state/region, etc.
        "LOCATION",     # Other location labels
        "ORG",          # Organizations (e.g., hospitals, clinics, employers)
        "DATE",         # Dates (DOB, admission date, etc.)
        "TIME",         # Times, if your model distinguishes them separately
        "CONTACT",      # Could include phone, fax, email, etc.
        "ID",           # Could include SSN, MRN, account, license, etc.
        "URL",          # Websites or IP addresses
        "FAC",          # Facility names (some models use FAC for “facility”)
        # "PRODUCT",      # Sometimes used for device identifiers
        "LICENSE",      # Certificate/license numbers (if your model uses this label)
    ]

    # 3) Analyze
    summary, details = analyze_maskers_across_texts(
        texts=texts,
        maskers=maskers,
        masker_names=masker_names,
        entity_types=hipaa_ner_labels
    )

    # 6) (Optional) Print out the details for each text
    print("\n=== PER-TEXT DETAILS (optional) ===")
    for result in details[:100]:
        print(f"\n[Text ID: {result['text_id']}] {result['text']}")
        for mk, spans in result["masker_spans"].items():
            print(f"  {mk} => {spans}")
        print("  Pairwise overlaps =>", result["pairwise_overlaps"])

    # 4) Print summary results
    print("=== SUMMARY ACROSS ALL TEXTS ===\n")
    print("Total Spans per Masker:")
    for mk, count in summary["total_spans_per_masker"].items():
        print(f"  {mk}: {count}")

    print("\nTotal Pairwise Overlaps:")
    for mk_pair, overlap_val in summary["total_pairwise_overlaps"].items():
        print(f"  {mk_pair}: {overlap_val}")

    # 5) Show example usage of redundancy check
    if len(masker_names) > 1:
        # Let's just compare the first two for demonstration
        main_masker = masker_names[0]
        other_masker = masker_names[1]
        is_redundant = decide_if_masker_is_redundant(summary, main_masker, other_masker)
        if is_redundant:
            print(f"\n{other_masker} is fully covered by {main_masker}!")
        else:
            print(f"\n{other_masker} is NOT fully covered by {main_masker}.")


if __name__ == "__main__":
    main()

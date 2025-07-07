import re
import os
import spacy
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import unicodedata
import json
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, partial
import multiprocessing as mp
from typing import List, Tuple, Set, Optional, Union
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

#######################################################################
#  Optimized Maskers
#######################################################################

class OptimizedRegexMasker:
    """
    Heavily optimized RegEx-based masker with pre-compiled patterns,
    vectorized operations, and intelligent caching.
    """

    def __init__(self, known_names=None, debug=False):
        self.debug = debug
        self.known_names = set(known_names or [])

        # Pre-compile all regex patterns for better performance
        self._compiled_patterns = [
            (re.compile(r"\b(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b"), "PERSON"),
            (re.compile(r"(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])[-/]((?:19|20)\d{2}|\d{2})"), "DATE"),
            (re.compile(r"\d{3}-\d{2}-\d{4}"), "SSN"),
            (re.compile(r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?=[^0-9]|$)"), "PHONE"),
            (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}(?=[^a-zA-Z0-9._%+\-]|$)"), "EMAIL"),
            (re.compile(r"(?:MRN\s*\d{5,}|\d{7,})"), "MRN"),
            (re.compile(r"\d{1,5}\s+(?:[A-Za-z0-9]+\s?){1,6},\s?[A-Za-z]+(?:[ -][A-Za-z]+)*,\s?[A-Z]{2},\s?\d{5}"), "ADDRESS"),
        ]

        # Pre-compile token extraction pattern
        self._token_pattern = re.compile(r"[A-Za-z]+")
        
        # Cache for normalized text to avoid repeated normalization
        self._normalization_cache = {}

    @lru_cache(maxsize=1000)
    def _normalize_text(self, text: str) -> str:
        """Cached text normalization."""
        return unicodedata.normalize("NFKC", text)

    def _validate_person_tokens(self, matched_str: str) -> bool:
        """Optimized person name validation using cached token extraction."""
        if not self.known_names:
            return True
        
        tokens = self._token_pattern.findall(matched_str)
        return any(token in self.known_names for token in tokens)

    def get_entity_spans(self, text: str, entity_types: Optional[Union[str, List[str]]] = None) -> List[Tuple[int, int, str]]:
        """
        Optimized entity span detection with early filtering and reduced allocations.
        """
        if isinstance(entity_types, str):
            entity_types = [entity_types]

        # Convert numpy.str_ to regular str if needed
        if hasattr(text, 'item'):
            text = text.item()
        elif not isinstance(text, str):
            text = str(text)

        entity_types_set = set(entity_types) if entity_types else None
        found_spans = []
        normalized_text = self._normalize_text(text)

        for pattern, label in self._compiled_patterns:
            if entity_types_set and label not in entity_types_set:
                continue

            for match in pattern.finditer(normalized_text):
                start, end = match.start(), match.end()
                
                if label == "PERSON":
                    matched_str = normalized_text[start:end]
                    if not self._validate_person_tokens(matched_str):
                        continue

                found_spans.append((start, end, label))
                if self.debug:
                    print(f"[DEBUG] RegexMasker Matched '{normalized_text[start:end]}' as {label}")
                    
        return found_spans


class OptimizedHuggingfaceMasker:
    """
    Optimized Hugging Face masker with batch processing and model caching.
    """
    
    def __init__(self, model_name="StanfordAIMI/stanford-deidentifier-only-i2b2", cache_dir=None, device=0, batch_size=32):
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Use faster tokenizer settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            use_fast=True,  # Use fast tokenizer
            model_max_length=512  # Limit token length for faster processing
        )
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            torch_dtype='auto'  # Use automatic mixed precision when available
        )
        
        effective_device = device if device >= 0 else -1
        # Create pipeline without return_all_scores (not supported in some versions)
        self.nlp = pipeline(
            "ner", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            device=effective_device, 
            aggregation_strategy="simple",
            batch_size=self.batch_size  # Enable batching
        )

    def get_entity_spans(self, text: str, entity_types: Optional[Union[str, List[str]]] = None) -> List[Tuple[int, int, str]]:
        if not text.strip():
            return []

        if isinstance(entity_types, str):
            entity_types = [entity_types]

        # Convert numpy.str_ to regular str if needed
        if hasattr(text, 'item'):
            text = text.item()
        elif not isinstance(text, str):
            text = str(text)

        # Truncate text if too long to avoid memory issues
        if len(text) > 10000:
            text = text[:10000]

        try:
            ner_results = self.nlp(text)
        except Exception as e:
            print(f"Warning: HuggingFace NER failed: {e}")
            return []

        spans = []
        entity_types_lower = [et.lower() for et in entity_types] if entity_types else None
        
        for entity in ner_results:
            label = entity['entity_group']

            if entity_types_lower and not any(e_type in label.lower() for e_type in entity_types_lower):
                continue
                
            spans.append((entity["start"], entity["end"], label))
            
        return spans


class OptimizedSpaCyNERMasker:
    """
    Optimized SpaCy masker with disabled unnecessary pipeline components.
    """
    
    def __init__(self, model_name="en_core_web_trf"):
        try:
            if spacy.prefer_gpu():
                print(f"SpaCy attempting GPU for '{model_name}'.")
            else:
                print(f"SpaCy using CPU for '{model_name}'.")
        except Exception as e:
            print(f"SpaCy GPU check failed: {e}")
        
        # Load model with only necessary components for maximum speed
        self.nlp = spacy.load(
            model_name,
            disable=["parser", "tagger", "lemmatizer", "attribute_ruler", "tok2vec"]  # Disable unnecessary components
        )
        
        # Enable only NER
        if "ner" not in self.nlp.pipe_names:
            print(f"Warning: NER component not found in {model_name}")

    def get_entity_spans(self, text: str, entity_types: Optional[Union[str, List[str]]] = None) -> List[Tuple[int, int, str]]:
        if not text.strip():
            return []

        if isinstance(entity_types, str):
            entity_types = [entity_types]

        # Convert numpy.str_ to regular str if needed
        if hasattr(text, 'item'):
            text = text.item()
        elif not isinstance(text, str):
            text = str(text)

        # Process with optimized pipeline
        doc = self.nlp(text)
        entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

        if entity_types:
            entity_types_lower = [et.lower() for et in entity_types]
            entities = [
                (start, end, label) for start, end, label in entities
                if any(e_type in label.lower() for e_type in entity_types_lower)
            ]
            
        return entities

#######################################################################
# Optimized Analysis Functions
#######################################################################

@lru_cache(maxsize=1)
def load_names_cached(file_path: str = "./data/names.json") -> Set[str]:
    """
    Cached version of load_names that only loads once.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Names file not found at {file_path}")
        return set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            names_dict = json.load(f)
        
        combined = (
            names_dict.get("women", []) +
            names_dict.get("men", []) +
            names_dict.get("last", [])
        )
        return set(combined)
    except Exception as e:
        print(f"Error loading names from {file_path}: {e}")
        return set()


def vectorized_replace_consecutive_tags(texts: Union[str, pd.Series], mask_tag: str) -> Union[str, pd.Series]:
    """
    Vectorized version for pandas Series or single string processing.
    """
    escaped_mask_tag = re.escape(mask_tag)
    pattern = re.compile(r'(' + escaped_mask_tag + r')(?:\s*\1)+')
    
    if isinstance(texts, pd.Series):
        return texts.str.replace(pattern, r'\1', regex=True)
    else:
        return pattern.sub(r'\1', texts)


def optimized_merge_spans(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Optimized span merging using numpy for better performance on large datasets.
    """
    if not spans:
        return []
    
    if len(spans) == 1:
        return spans
    
    # Convert to numpy array for faster operations
    spans_array = np.array([(s[0], s[1]) for s in spans])
    labels = [s[2] for s in spans]
    
    # Sort by start position
    sort_indices = np.argsort(spans_array[:, 0])
    sorted_spans = spans_array[sort_indices]
    sorted_labels = [labels[i] for i in sort_indices]
    
    merged = []
    current_start, current_end = sorted_spans[0]
    current_label = sorted_labels[0]

    for i in range(1, len(sorted_spans)):
        next_start, next_end = sorted_spans[i]
        next_label = sorted_labels[i]

        if next_start < current_end:
            current_end = max(current_end, next_end)
        else:
            merged.append((int(current_start), int(current_end), current_label))
            current_start, current_end, current_label = next_start, next_end, next_label
    
    merged.append((int(current_start), int(current_end), current_label))
    return merged


def optimized_mask_text(text: str, spans: List[Tuple[int, int, str]], mask: str = '[REDACTED]') -> str:
    """
    Optimized text masking using list pre-allocation and efficient string building.
    """
    if not spans:
        return text
    
    # Pre-allocate result list for better performance
    result_parts = []
    last_end = 0
    
    for start, end, label in spans:
        if start < last_end:
            continue  # Skip overlapping spans
        
        result_parts.append(text[last_end:start])
        result_parts.append(mask)
        last_end = end
    
    result_parts.append(text[last_end:])
    return "".join(result_parts)


class OptimizedDeidentifier:
    """
    Highly optimized deidentifier with parallel processing, caching, and vectorization.
    """
    
    def __init__(self, maskers: List, default_mask: str = '[REDACTED]', n_jobs: int = None):
        self.maskers = maskers
        self.default_mask = default_mask
        self.n_jobs = n_jobs or min(mp.cpu_count(), 4)  # Limit to 4 to avoid overwhelming system
        
        # Pre-compile mask replacement pattern
        self._mask_pattern = re.compile(r'(' + re.escape(default_mask) + r')(?:\s*\1)+')
        
        # Set up tqdm for pandas
        tqdm.pandas(desc="De-identifying")

    def _process_single_text(self, text: str, entity_types: Optional[List[str]] = None) -> str:
        """
        Optimized single text processing with early returns and minimal allocations.
        """
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            return text

        all_spans = []
        
        # Process maskers sequentially to avoid memory overhead
        for masker_instance in self.maskers:
            try:
                spans = masker_instance.get_entity_spans(text, entity_types=entity_types)
                all_spans.extend(spans)
            except Exception as e:
                print(f"Error in {type(masker_instance).__name__}: {e}")
                continue

        if not all_spans:
            return text

        # Filter spans efficiently
        filtered_spans = [
            (start, end, label) for start, end, label in all_spans
            if not (text[start:end].isupper() and 'person' not in label.lower())
        ]

        if not filtered_spans:
            return text

        merged_spans = optimized_merge_spans(filtered_spans)
        masked_text = optimized_mask_text(text, merged_spans, mask=self.default_mask)
        
        # Apply consecutive tag replacement
        masked_text = self._mask_pattern.sub(r'\1', masked_text)
        
        return masked_text

    def deidentify(self, text: str, entity_types: Optional[List[str]] = None) -> str:
        """
        Single text deidentification.
        """
        return self._process_single_text(text, entity_types)

    def deidentify_batch(self, texts: List[str], entity_types: Optional[List[str]] = None) -> List[str]:
        """
        Batch processing with parallel execution for multiple texts.
        """
        if len(texts) == 1:
            return [self._process_single_text(texts[0], entity_types)]
        
        # Use parallel processing for batch operations
        process_func = partial(self._process_single_text, entity_types=entity_types)
        
        # Choose between threading and multiprocessing based on workload
        if len(texts) < 10:  # Small batch - use threading
            with ThreadPoolExecutor(max_workers=min(self.n_jobs, len(texts))) as executor:
                return list(executor.map(process_func, texts))
        else:  # Large batch - use multiprocessing
            with ProcessPoolExecutor(max_workers=min(self.n_jobs, len(texts))) as executor:
                return list(executor.map(process_func, texts))

    def deidentify_csv(self, input_csv_path: str, output_csv_path: str, column_name: str, 
                      entity_types: Optional[List[str]] = None, chunk_size: int = 1000):
        """
        Optimized CSV processing with chunking and vectorized operations.
        """
        try:
            # Process CSV in chunks for memory efficiency
            chunk_iter = pd.read_csv(input_csv_path, chunksize=chunk_size)
            first_chunk = True
            
            for chunk_df in tqdm(chunk_iter, desc="Processing CSV chunks"):
                if column_name not in chunk_df.columns:
                    print(f"Error: Column '{column_name}' not found. Available: {chunk_df.columns.tolist()}")
                    return

                # Prepare text data
                text_series = chunk_df[column_name].astype(str).fillna('')
                text_list = text_series.tolist()
                
                # Process in batch
                processed_texts = self.deidentify_batch(text_list, entity_types)
                chunk_df[column_name] = processed_texts
                
                # Write chunk to output
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                chunk_df.to_csv(output_csv_path, mode=mode, header=header, index=False)
                first_chunk = False
                
            print(f"De-identified data successfully saved to {output_csv_path}")
            
        except FileNotFoundError:
            print(f"Error: Input CSV file not found at {input_csv_path}")
        except Exception as e:
            print(f"Error processing CSV: {e}")

#######################################################################
# Main Entry Point
#######################################################################

def main():
    parser = argparse.ArgumentParser(description="Optimized de-identification tool for maximum performance.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="A single text string to de-identify.")
    group.add_argument("--input_csv", type=str, help="Path to the input CSV file for de-identification.")

    parser.add_argument("--column_name", type=str, help="Name of the column to de-identify in the CSV.")
    parser.add_argument("--output_csv", type=str, help="Path to save the de-identified CSV file.")
    parser.add_argument("--output_text_file", type=str, help="Path to save the de-identified text output.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for CSV processing (default: 1000)")

    parser.add_argument("--maskers", nargs='+', choices=['regex', 'spacy', 'huggingface'],
                        default=['regex'], help="List of maskers to use. Default: ['regex']")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf",
                        help="SpaCy model name. Default: en_core_web_trf")
    parser.add_argument("--hf_models", nargs='+', default=["StanfordAIMI/stanford-deidentifier-only-i2b2"],
                        help="Hugging Face model name(s).")
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="HF models cache directory.")
    parser.add_argument("--hf_device", type=int, default=0, help="HF device (0 for GPU, -1 for CPU).")
    parser.add_argument("--hf_batch_size", type=int, default=32, help="HF batch size for processing.")
    parser.add_argument("--names_file", type=str, default="./data/names.json", help="Path to names JSON file.")
    parser.add_argument("--regex_debug", action='store_true', help="Enable regex debug mode.")
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of parallel jobs (default: auto).")

    parser.add_argument("--entity_types", nargs='*', default=None,
                        help="Specific entity types to mask (e.g., PERSON ORG DATE).")
    parser.add_argument("--default_mask", type=str, default="[REDACTED]",
                        help="Replacement string for identified PHI. Default: '[REDACTED]'")

    args = parser.parse_args()

    if args.input_csv and not (args.output_csv and args.column_name):
        parser.error("--input_csv requires --output_csv and --column_name.")

    # Initialize optimized maskers
    active_maskers = []
    print("Initializing optimized maskers...")
    
    if "regex" in args.maskers:
        known_names = load_names_cached(args.names_file)
        active_maskers.append(OptimizedRegexMasker(known_names=known_names, debug=args.regex_debug))
        print(f"OptimizedRegexMasker initialized with {len(known_names)} known names.")
        
    if "spacy" in args.maskers:
        try:
            active_maskers.append(OptimizedSpaCyNERMasker(model_name=args.spacy_model))
            print(f"OptimizedSpaCyNERMasker initialized with model: {args.spacy_model}")
        except Exception as e:
            print(f"Error initializing SpaCy model '{args.spacy_model}': {e}")
            print("Ensure model is downloaded: python -m spacy download en_core_web_trf")
            return
            
    if "huggingface" in args.maskers:
        for model_name in args.hf_models:
            try:
                active_maskers.append(OptimizedHuggingfaceMasker(
                    model_name=model_name,
                    cache_dir=args.hf_cache_dir,
                    device=args.hf_device,
                    batch_size=args.hf_batch_size
                ))
                print(f"OptimizedHuggingfaceMasker initialized: {model_name}")
            except Exception as e:
                print(f"Error initializing HF model '{model_name}': {e}")
                return

    if not active_maskers:
        print("No maskers initialized successfully. Exiting.")
        return

    # Initialize optimized deidentifier
    deidentifier = OptimizedDeidentifier(
        active_maskers, 
        default_mask=args.default_mask,
        n_jobs=args.n_jobs
    )
    print(f"OptimizedDeidentifier initialized with {len(active_maskers)} masker(s), using {deidentifier.n_jobs} parallel jobs.")
    
    entity_types = args.entity_types
    if entity_types:
        print(f"Targeting entity types: {', '.join(entity_types)}")
    else:
        print("Using default entity detection scope.")

    # Process input
    if args.input_csv:
        print(f"\nProcessing CSV: {args.input_csv}")
        print(f"Column: '{args.column_name}', Chunk size: {args.chunk_size}")
        print(f"Output: {args.output_csv}")
        
        deidentifier.deidentify_csv(
            input_csv_path=args.input_csv,
            output_csv_path=args.output_csv,
            column_name=args.column_name,
            entity_types=entity_types,
            chunk_size=args.chunk_size
        )
    elif args.text:
        print(f"\nOriginal: \"{args.text}\"")
        masked_text = deidentifier.deidentify(args.text, entity_types=entity_types)
        print(f"Masked:   \"{masked_text}\"")
        
        if args.output_text_file:
            try:
                with open(args.output_text_file, 'w', encoding='utf-8') as f:
                    f.write(masked_text)
                print(f"Output saved to: {args.output_text_file}")
            except IOError as e:
                print(f"Error writing to {args.output_text_file}: {e}")

    print("\nOptimized de-identification complete.")


if __name__ == "__main__":
    main()
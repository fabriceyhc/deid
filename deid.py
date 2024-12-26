import re
import os
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class HuggingfaceMasker:
    def __init__(self, model_name="StanfordAIMI/stanford-deidentifier-only-i2b2", cache_dir="/data2/.shared_models"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def get_entity_spans(self, text, entity_type=None):
        """
        Return a list of (start, end, label) for each detected entity.
        If entity_type is provided, only return spans of that type.
        """
        ner_results = self.nlp(text)
        ner_results = sorted(ner_results, key=lambda x: x['start'])

        spans = []
        for entity in ner_results:
            label = entity['entity']
            if entity_type is not None and entity_type not in label:
                continue
            spans.append((entity["start"], entity["end"], label))
        return spans

class SpaCyNERMasker:
    def __init__(self, model_name="en_core_web_trf"):
        self.nlp = spacy.load(model_name)

    def get_entity_spans(self, text, entity_type=None):
        """
        Return a list of (start, end, label) for each detected entity.
        If entity_type is provided, only return spans of that type.
        Also removes 'PRODUCT' entities.
        """
        doc = self.nlp(text)
        entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

        # Filter by entity_type if specified
        if entity_type is not None:
            entities = [e for e in entities if entity_type in e[2]]

        # Remove PRODUCTS
        entities = [e for e in entities if "PRODUCT" not in e[2]]

        return entities

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

    def deidentify(self, text, entity_type=None):
        """
        Runs all maskers on the given text, merges identified PHI spans, and masks them out.
        If entity_type is specified, only mask entities of that type.
        """
        all_spans = []
        for m in self.maskers:
            spans = m.get_entity_spans(text, entity_type=entity_type)
            all_spans.extend(spans)

        # Merge overlapping spans from all models
        merged_spans = merge_spans(all_spans)

        # Mask the text once
        masked_text = mask_text(text, merged_spans, mask=self.default_mask)

        # Optionally replace consecutive tags if needed
        masked_text = replace_consecutive_NER_tags(masked_text)
        return masked_text

    def deidentify_csv(self, input_csv_path, output_csv_path, column_name, entity_type=None):
        """
        1. Reads a CSV file from input_csv_path
        2. Applies de-identification to 'column_name'
        3. Saves the de-identified data to output_csv_path
        
        :param input_csv_path: Path to the input CSV file
        :param output_csv_path: Path where the output CSV file will be saved
        :param column_name: The name of the column in which we have the text to de-identify
        :param entity_type: If specified, only mask entities matching this label 
                            (e.g., "PERSON"). Otherwise, mask all detected entities.
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
            return self.deidentify(text, entity_type=entity_type)

        df[column_name] = df[column_name].apply(deid_func)

        # Save the updated DataFrame to a new CSV
        df.to_csv(output_csv_path, index=False)

#############################
# Example usage
#############################

if __name__ == "__main__":

    # CUDA_VISIBLE_DEVICES=0 python -m deid

    spacy_masker = SpaCyNERMasker(model_name="en_core_web_trf")
    hf_masker_1 = HuggingfaceMasker(model_name="StanfordAIMI/stanford-deidentifier-base")
    hf_masker_2 = HuggingfaceMasker(model_name="obi/deid_roberta_i2b2")

    # Instantiate the Deidentifier class with a list of maskers
    deidentifier = Deidentifier([spacy_masker, hf_masker_1, hf_masker_2])

    # examples = [
    #     "Fabrice denies snorting Klonopin, but Hritika is skeptical.",
    #     "Panayiotis denies snorting Xanax, but DIVYANSH is skeptical.",
    #     "Patient John Doe (DOB: 01/01/1980) was admitted with symptoms of severe anxiety and was prescribed Xanax for immediate relief.",
    #     "On 03/15/2024, Jane Smith was seen by Dr. Emily Johnson at Lakeside Medical Center for a follow-up regarding her chronic insomnia. Ambien dosage was adjusted.",
    #     "Michael Brown, a 45-year-old with a history of type 2 diabetes, was advised to continue with their current medication, Metformin, and to monitor their blood sugar levels closely.",
    #     "The blood test results from 04/10/2024 for Alex Martinez indicate a need to modify the current treatment plan to include Lipitor for better cholesterol management.",
    #     "Sarah Wilson's insurance provider, HealthFirst Insurance, has approved coverage for the prescribed Humira starting from 05/01/2024.",
    #     "During the consultation on 02/20/2024, Lisa Chang disclosed a family history of glaucoma. Considering this, Timolol eye drops were prescribed as a precautionary measure.",
    #     "Kevin Lee underwent a successful laparoscopic cholecystectomy on 01/25/2024 and is scheduled for a follow-up on 02/15/2024 to assess the need for further medication adjustments, including the possibility of incorporating Ursodiol.",
    # ]

    # for i, ex in enumerate(examples, 1):
    #     masked = deidentifier.deidentify(ex)
    #     print(f"Example {i}:\nOriginal: {ex}\nMasked:   {masked}\n")

    base_path = "F:/Inbound/CTSI/CLShover_24_22-001273/Data"
    
    PHI_candidates = [
        ("Problem_Lists.csv", "PROBLEM_DESCRIPTION"),
        ("Labs.csv", "RESULT"),
        ("Social_History.csv", "ALCOHOL_COMMENTS"),
        ("Social_History.csv", "ILLICIT_DRUG_COMMENTS"),
        ("Provider_Notes.csv", "NOTE_TEXT"),
    ]

    for filename, column in PHI_candidates:
        deidentifier.deidentify_csv(
            input_csv_path=os.path.join(base_path, filename),
            output_csv_path=os.path.join(base_path, filename.replace(".csv", "_deid.csv")),
            column_name=column,
        )

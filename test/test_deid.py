import unittest
import pandas as pd
import os
import json
from deid import Deidentifier, RegexMasker, HuggingfaceMasker, SpaCyNERMasker, load_names, merge_spans, mask_text, replace_consecutive_NER_tags

class TestHelperFunctions(unittest.TestCase):

    def test_load_names(self):
        # Create a dummy names file
        names_data = {"men": ["John", "David"], "women": ["Jane"], "last": ["Smith", "Doe"]}
        names_file_path = "test_names.json"
        with open(names_file_path, 'w') as f:
            json.dump(names_data, f)

        # Test loading as set
        names_set = load_names(names_file_path, as_set=True)
        self.assertEqual(names_set, {"John", "David", "Jane", "Smith", "Doe"})

        # Test loading as list
        names_list = load_names(names_file_path, as_set=False)
        self.assertCountEqual(names_list, ["John", "David", "Jane", "Smith", "Doe"])

        # Test file not found
        os.remove(names_file_path)
        names_set_empty = load_names("non_existent_file.json")
        self.assertEqual(names_set_empty, set())

    def test_merge_spans(self):
        spans = [(10, 20, 'PERSON'), (15, 25, 'PERSON'), (30, 40, 'DATE')]
        merged = merge_spans(spans)
        self.assertEqual(merged, [(10, 25, 'PERSON'), (30, 40, 'DATE')])

        spans_no_overlap = [(10, 20, 'PERSON'), (30, 40, 'DATE')]
        merged = merge_spans(spans_no_overlap)
        self.assertEqual(merged, spans_no_overlap)

        self.assertEqual(merge_spans([]), [])

    def test_mask_text(self):
        text = "John Smith lives in New York."
        spans = [(0, 10, 'PERSON'), (22, 31, 'GPE')]
        masked = mask_text(text, spans, mask='[REDACTED]')
        self.assertEqual(masked, '[REDACTED] lives in [REDACTED].')

    def test_replace_consecutive_ner_tags(self):
        text = "[REDACTED] [REDACTED] was seen on [REDACTED]."
        result = replace_consecutive_NER_tags(text, '[REDACTED]')
        self.assertEqual(result, "[REDACTED] was seen on [REDACTED].")

class TestDeidentifier(unittest.TestCase):

    def setUp(self):
        # This method will be called before each test
        self.known_names = ["John", "Doe", "Smith"]
        self.regex_masker = RegexMasker(known_names=self.known_names)
        self.spacy_masker = SpaCyNERMasker()
        self.huggingface_masker = HuggingfaceMasker()
        self.deidentifier = Deidentifier(maskers=[self.regex_masker, self.spacy_masker, self.huggingface_masker])

    def test_deidentifier_instantiation(self):
        deidentifier = Deidentifier(maskers=[])
        self.assertIsNotNone(deidentifier)

    def test_regex_masker_names(self):
        text = "A patient named John Doe visited."
        spans = self.regex_masker.get_entity_spans(text)
        self.assertIn((16, 25, 'PERSON'), spans)

    def test_regex_masker_phone(self):
        text = "Call 555-123-4567 for details."
        spans = self.regex_masker.get_entity_spans(text)
        self.assertIn((5, 17, 'PHONE'), spans)

    def test_regex_masker_email(self):
        text = "Email is test.person@example.com."
        spans = self.regex_masker.get_entity_spans(text)
        self.assertIn((9, 34, 'EMAIL'), spans)

    def test_spacy_masker(self):
        text = "Dr. Smith works at Mass General."
        spans = self.spacy_masker.get_entity_spans(text)
        self.assertTrue(any(label in ['PERSON', 'ORG'] for _, _, label in spans))

    def test_huggingface_masker(self):
        text = "The patient is Jane Doe."
        spans = self.huggingface_masker.get_entity_spans(text)
        self.assertTrue(any(label in ['PER', 'PERSON'] for _, _, label in spans))

    def test_deidentify_text(self):
        text = "Patient John Smith, phone 555-123-4567."
        masked_text = self.deidentifier.deidentify(text)
        self.assertEqual(masked_text, "Patient [REDACTED], phone [REDACTED].")

    def test_deidentify_csv(self):
        # Create a dummy CSV in memory
        input_data = {'id': [1], 'text': ["Patient Johnathan Michael Smith (DOB: 01/15/1975, MRN: 1234567) was seen on 03/10/2024 by Dr. Emily White."]}
        input_df = pd.DataFrame(input_data)
        
        input_csv_path = "temp_input_test.csv"
        output_csv_path = "temp_output_test.csv"
        
        input_df.to_csv(input_csv_path, index=False)

        # De-identify the CSV
        self.deidentifier.deidentify_csv(input_csv_path, output_csv_path, "text")

        # Check the output
        output_df = pd.read_csv(output_csv_path)
        self.assertIn("[REDACTED]", output_df['text'][0])
        
        # Cleanup
        os.remove(input_csv_path)
        os.remove(output_csv_path)

if __name__ == '__main__':

    # RUN: python -m unittest discover -s test

    unittest.main()
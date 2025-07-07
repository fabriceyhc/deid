#!/usr/bin/env python3
"""
Comprehensive Benchmark: Tests speed and correctness of de-identification maskers
using real test data and comparing against expected output.
"""

import time
import pandas as pd
import os
import sys
from typing import List, Dict, Tuple
import difflib
import re

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deid import (
    OptimizedRegexMasker, 
    OptimizedSpaCyNERMasker,
    OptimizedHuggingfaceMasker,
    OptimizedDeidentifier,
    load_names_cached
)


class BenchmarkResults:
    """Container for benchmark results."""
    
    def __init__(self, name: str):
        self.name = name
        self.success = False
        self.processing_time = 0.0
        self.initialization_time = 0.0
        self.total_time = 0.0
        self.texts_per_second = 0.0
        self.correctness_score = 0.0
        self.exact_matches = 0
        self.total_texts = 0
        self.output_file = ""
        self.error_message = ""
        self.redaction_coverage = 0.0  # % of expected redactions found
        self.over_redaction = 0.0      # % of extra redactions made


def load_test_data(input_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load test data from CSV file."""
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Test data not found: {input_path}")
    
    df = pd.read_csv(input_path)
    if 'text' not in df.columns:
        raise ValueError("Test data must have a 'text' column")
    
    texts = df['text'].astype(str).tolist()
    print(f"✓ Loaded {len(texts)} test texts from {input_path}")
    
    return df, texts


def load_expected_output(expected_path: str) -> List[str]:
    """Load expected de-identified output."""
    
    if not os.path.exists(expected_path):
        print(f"⚠ Expected output not found: {expected_path}")
        return []
    
    df = pd.read_csv(expected_path)
    if 'text' not in df.columns:
        raise ValueError("Expected output must have a 'text' column")
    
    expected_texts = df['text'].astype(str).tolist()
    print(f"✓ Loaded {len(expected_texts)} expected outputs from {expected_path}")
    
    return expected_texts


def calculate_correctness_metrics(actual: List[str], expected: List[str]) -> Tuple[float, float, int, float, float]:
    """
    Calculate correctness metrics by comparing actual vs expected output.
    
    Returns:
        - overall_score: Weighted correctness score (0-100)
        - exact_matches: Number of texts that match exactly
        - redaction_coverage: % of expected redactions found
        - over_redaction: % of extra redactions made
    """
    
    if not expected:
        # No expected output to compare against
        return 0.0, 0, 0.0, 0.0
    
    if len(actual) != len(expected):
        print(f"⚠ Length mismatch: actual={len(actual)}, expected={len(expected)}")
        min_len = min(len(actual), len(expected))
        actual = actual[:min_len]
        expected = expected[:min_len]
    
    exact_matches = 0
    total_similarity = 0.0
    total_expected_redactions = 0
    total_found_redactions = 0
    total_actual_redactions = 0
    
    for i, (actual_text, expected_text) in enumerate(zip(actual, expected)):
        # Exact match check
        if actual_text.strip() == expected_text.strip():
            exact_matches += 1
            total_similarity += 100.0
        else:
            # Calculate similarity using sequence matcher
            similarity = difflib.SequenceMatcher(None, actual_text, expected_text).ratio() * 100
            total_similarity += similarity
        
        # Count redactions
        expected_redactions = expected_text.count('[REDACTED]')
        actual_redactions = actual_text.count('[REDACTED]')
        
        total_expected_redactions += expected_redactions
        total_actual_redactions += actual_redactions
        
        # Count how many expected redaction positions were found
        # This is a simplified check - in reality, we'd need more sophisticated alignment
        if expected_redactions > 0:
            found_redactions = min(actual_redactions, expected_redactions)
            total_found_redactions += found_redactions
    
    # Calculate metrics
    overall_score = total_similarity / len(actual) if actual else 0.0
    redaction_coverage = (total_found_redactions / total_expected_redactions * 100) if total_expected_redactions > 0 else 0.0
    over_redaction = max(0, (total_actual_redactions - total_expected_redactions) / max(1, total_expected_redactions) * 100)
    
    return overall_score, exact_matches, redaction_coverage, over_redaction


def benchmark_masker_configuration(
    config_name: str,
    maskers: List,
    input_df: pd.DataFrame,
    texts: List[str],
    expected_output: List[str],
    output_dir: str
) -> BenchmarkResults:
    """Benchmark a specific masker configuration."""
    
    print(f"\n{'='*60}")
    print(f"🔬 Testing: {config_name}")
    print(f"{'='*60}")
    
    results = BenchmarkResults(config_name)
    results.total_texts = len(texts)
    
    try:
        # Initialize maskers
        start_init = time.time()
        deidentifier = OptimizedDeidentifier(maskers, n_jobs=1)  # Single thread for consistent timing
        results.initialization_time = time.time() - start_init
        
        print(f"✓ Initialized {len(maskers)} masker(s) in {results.initialization_time:.3f}s")
        
        # Process texts
        start_proc = time.time()
        processed_texts = []
        
        for text in texts:
            processed_text = deidentifier.deidentify(text)
            processed_texts.append(processed_text)
        
        results.processing_time = time.time() - start_proc
        results.total_time = results.initialization_time + results.processing_time
        
        # Calculate speed metrics
        results.texts_per_second = len(texts) / results.processing_time if results.processing_time > 0 else 0
        
        print(f"✓ Processed {len(texts)} texts in {results.processing_time:.3f}s")
        print(f"✓ Rate: {results.texts_per_second:.1f} texts/second")
        
        # Save output
        output_file = os.path.join(output_dir, f"test_deid_{config_name.lower().replace(' ', '_')}.csv")
        output_df = input_df.copy()
        output_df['text'] = processed_texts
        output_df.to_csv(output_file, index=False)
        results.output_file = output_file
        
        print(f"✓ Saved output to: {output_file}")
        
        # Calculate correctness metrics
        if expected_output:
            correctness, exact, coverage, over_redact = calculate_correctness_metrics(
                processed_texts, expected_output
            )
            results.correctness_score = correctness
            results.exact_matches = exact
            results.redaction_coverage = coverage
            results.over_redaction = over_redact
            
            print(f"✓ Correctness score: {correctness:.1f}%")
            print(f"✓ Exact matches: {exact}/{len(texts)} ({exact/len(texts)*100:.1f}%)")
            print(f"✓ Redaction coverage: {coverage:.1f}%")
            print(f"✓ Over-redaction: {over_redact:.1f}%")
            
            # Show first example
            if processed_texts and expected_output:
                print(f"\n📝 Example comparison:")
                print(f"  Original:  '{texts[0][:80]}...'")
                print(f"  Actual:    '{processed_texts[0][:80]}...'")
                print(f"  Expected:  '{expected_output[0][:80]}...'")
        else:
            print(f"⚠ No expected output for correctness comparison")
        
        results.success = True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        results.error_message = str(e)
        results.success = False
        import traceback
        traceback.print_exc()
    
    return results


def create_comparison_report(all_results: List[BenchmarkResults], output_dir: str):
    """Create a comprehensive comparison report."""
    
    report_path = os.path.join(output_dir, "benchmark_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# De-identification Benchmark Report\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Performance Summary
        f.write("## 📊 Performance Summary\n\n")
        f.write("| Configuration | Status | Speed (texts/s) | Correctness (%) | Exact Matches | Redaction Coverage | Over-redaction |\n")
        f.write("|---------------|--------|-----------------|-----------------|---------------|-------------------|----------------|\n")
        
        for result in all_results:
            if result.success:
                status = "✅ PASS"
                speed = f"{result.texts_per_second:.1f}"
                correctness = f"{result.correctness_score:.1f}"
                exact = f"{result.exact_matches}/{result.total_texts}"
                coverage = f"{result.redaction_coverage:.1f}%"
                over_redact = f"{result.over_redaction:.1f}%"
            else:
                status = "❌ FAIL"
                speed = "N/A"
                correctness = "N/A"
                exact = "N/A"
                coverage = "N/A"
                over_redact = "N/A"
            
            f.write(f"| {result.name} | {status} | {speed} | {correctness} | {exact} | {coverage} | {over_redact} |\n")
        
        # Performance Rankings
        successful_results = [r for r in all_results if r.success]
        
        if successful_results:
            f.write("\n## 🏆 Performance Rankings\n\n")
            
            # Speed ranking
            speed_ranking = sorted(successful_results, key=lambda x: x.texts_per_second, reverse=True)
            f.write("### Speed Ranking (texts/second)\n")
            for i, result in enumerate(speed_ranking, 1):
                f.write(f"{i}. **{result.name}**: {result.texts_per_second:.1f} texts/s\n")
            
            # Correctness ranking
            correctness_ranking = sorted(successful_results, key=lambda x: x.correctness_score, reverse=True)
            f.write("\n### Correctness Ranking (%)\n")
            for i, result in enumerate(correctness_ranking, 1):
                f.write(f"{i}. **{result.name}**: {result.correctness_score:.1f}%\n")
            
            # Best overall (balanced score)
            f.write("\n### Balanced Score (Speed × Correctness)\n")
            for result in successful_results:
                balanced_score = result.texts_per_second * result.correctness_score / 100
                result.balanced_score = balanced_score
            
            balanced_ranking = sorted(successful_results, key=lambda x: x.balanced_score, reverse=True)
            for i, result in enumerate(balanced_ranking, 1):
                f.write(f"{i}. **{result.name}**: {result.balanced_score:.1f} (speed×correctness)\n")
        
        # Detailed Results
        f.write("\n## 📋 Detailed Results\n\n")
        for result in all_results:
            f.write(f"### {result.name}\n")
            if result.success:
                f.write(f"- **Status**: ✅ Success\n")
                f.write(f"- **Processing Time**: {result.processing_time:.3f}s\n")
                f.write(f"- **Speed**: {result.texts_per_second:.1f} texts/second\n")
                f.write(f"- **Correctness**: {result.correctness_score:.1f}%\n")
                f.write(f"- **Exact Matches**: {result.exact_matches}/{result.total_texts}\n")
                f.write(f"- **Output File**: `{os.path.basename(result.output_file)}`\n")
            else:
                f.write(f"- **Status**: ❌ Failed\n")
                f.write(f"- **Error**: {result.error_message}\n")
            f.write("\n")
        
        # Recommendations
        f.write("## 🎯 Recommendations\n\n")
        if successful_results:
            fastest = max(successful_results, key=lambda x: x.texts_per_second)
            most_accurate = max(successful_results, key=lambda x: x.correctness_score)
            most_balanced = max(successful_results, key=lambda x: x.balanced_score)
            
            f.write(f"- **Fastest Processing**: {fastest.name} ({fastest.texts_per_second:.1f} texts/s)\n")
            f.write(f"- **Highest Accuracy**: {most_accurate.name} ({most_accurate.correctness_score:.1f}%)\n")
            f.write(f"- **Best Balanced**: {most_balanced.name} (score: {most_balanced.balanced_score:.1f})\n")
            
        f.write("\n## 📁 Output Files\n\n")
        f.write("All de-identified outputs have been saved to:\n")
        for result in all_results:
            if result.success and result.output_file:
                f.write(f"- `{os.path.basename(result.output_file)}`\n")
    
    print(f"\n📊 Comprehensive report saved to: {report_path}")
    return report_path


def main():
    """Run comprehensive benchmark with speed and correctness testing."""
    
    print("🚀 Comprehensive De-identification Benchmark")
    print("Testing speed and correctness against real test data")
    print("=" * 70)
    
    # Paths
    input_path = "/home/fabrice/deid/test/data/unprocessed/test.csv"
    expected_path = "/home/fabrice/deid/test/data/processed/test_deid.csv"
    output_dir = "/home/fabrice/deid/test/data/processed"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load test data
        input_df, texts = load_test_data(input_path)
        expected_output = load_expected_output(expected_path)
        
        # Load known names for regex masker
        try:
            known_names = load_names_cached("./data/names.json")
            print(f"✓ Loaded {len(known_names)} known names")
        except:
            known_names = set()
            print("⚠ Using empty known names set")
        
        # Test configurations
        configurations = [
            {
                'name': 'Regex Only',
                'maskers': [OptimizedRegexMasker(known_names=known_names, debug=False)]
            },
            {
                'name': 'SpaCy Only',
                'maskers': [OptimizedSpaCyNERMasker(model_name="en_core_web_trf")]
            },
            {
                'name': 'HuggingFace Only',
                'maskers': [OptimizedHuggingfaceMasker(
                    model_name="StanfordAIMI/stanford-deidentifier-only-i2b2",
                    device=-1,  # CPU for consistency
                    batch_size=4
                )]
            },
            {
                'name': 'All Three Combined',
                'maskers': [
                    OptimizedRegexMasker(known_names=known_names, debug=False),
                    OptimizedSpaCyNERMasker(model_name="en_core_web_trf"),
                    OptimizedHuggingfaceMasker(
                        model_name="StanfordAIMI/stanford-deidentifier-only-i2b2",
                        device=-1,
                        batch_size=4
                    )
                ]
            }
        ]
        
        # Run benchmarks
        all_results = []
        
        for config in configurations:
            result = benchmark_masker_configuration(
                config['name'],
                config['maskers'],
                input_df,
                texts,
                expected_output,
                output_dir
            )
            all_results.append(result)
        
        # Create summary
        print(f"\n📊 FINAL SUMMARY")
        print("=" * 70)
        print(f"{'Configuration':<20} {'Status':<8} {'Speed':<12} {'Correctness':<12} {'Exact':<8}")
        print("-" * 70)
        
        for result in all_results:
            if result.success:
                status = "✅ PASS"
                speed = f"{result.texts_per_second:.1f}/s"
                correctness = f"{result.correctness_score:.1f}%"
                exact = f"{result.exact_matches}/{result.total_texts}"
            else:
                status = "❌ FAIL"
                speed = "N/A"
                correctness = "N/A"
                exact = "N/A"
            
            print(f"{result.name:<20} {status:<8} {speed:<12} {correctness:<12} {exact:<8}")
        
        # Create detailed report
        report_path = create_comparison_report(all_results, output_dir)
        
        print(f"\n🎉 Benchmark Complete!")
        print(f"📄 Detailed report: {report_path}")
        print(f"📁 Output files saved to: {output_dir}")
        
        # Show best performers
        successful = [r for r in all_results if r.success]
        if successful:
            fastest = max(successful, key=lambda x: x.texts_per_second)
            most_accurate = max(successful, key=lambda x: x.correctness_score)
            
            print(f"\n🏆 Best Performers:")
            print(f"   Fastest: {fastest.name} ({fastest.texts_per_second:.1f} texts/s)")
            print(f"   Most Accurate: {most_accurate.name} ({most_accurate.correctness_score:.1f}%)")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
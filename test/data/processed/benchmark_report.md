# De-identification Benchmark Report

Generated on: 2025-07-11 17:52:11

## 📊 Performance Summary

| Configuration | Status | Speed (texts/s) | Accuracy (%) | F1 Score (%) | Over-redaction Ratio | Exact Matches |
|---------------|--------|-----------------|--------------|--------------|---------------------|---------------|
| Regex Only | ✅ PASS | 111.7 | 99.5 | 92.6 | 1.15x | 0/8 |
| SpaCy Only | ✅ PASS | 0.2 | 100.0 | 51.2 | 2.91x | 0/8 |
| HuggingFace Only | ✅ PASS | 1.2 | 38.6 | 55.4 | 1.0x | 0/8 |
| All Three Combined | ✅ PASS | 0.2 | 100.0 | 44.4 | 3.51x | 0/8 |

## 🏆 Performance Rankings

### Speed Ranking (texts/second)
1. **Regex Only**: 111.7 texts/s
2. **HuggingFace Only**: 1.2 texts/s
3. **SpaCy Only**: 0.2 texts/s
4. **All Three Combined**: 0.2 texts/s

### Accuracy Ranking (% of expected redactions found)
1. **SpaCy Only**: 100.0%
2. **All Three Combined**: 100.0%
3. **Regex Only**: 99.5%
4. **HuggingFace Only**: 38.6%

### F1 Score Ranking (balanced accuracy & precision)
1. **Regex Only**: 92.6%
2. **HuggingFace Only**: 55.4%
3. **SpaCy Only**: 51.2%
4. **All Three Combined**: 44.4%

### Balanced Score (Speed × Accuracy)
1. **Regex Only**: 111.1 (speed×accuracy)
2. **HuggingFace Only**: 0.5 (speed×accuracy)
3. **SpaCy Only**: 0.2 (speed×accuracy)
4. **All Three Combined**: 0.2 (speed×accuracy)

## 📏 Scoring Metrics Explained

### Accuracy (0-100%)
- **Percentage of expected redactions that were actually found**
- **Higher is better** - shows how well the masker catches sensitive information
- **Combined maskers should score higher** as they catch more types of sensitive data

### F1 Score (0-100%)
- **Balanced metric** combining accuracy (recall) and precision
- **Accounts for over-redaction** - rewards finding expected redactions without excessive extras
- **Best metric for overall evaluation** when you want balance between thoroughness and precision

### Over-redaction Ratio
- **How many times more redactions were made than expected**
- **1.0x = perfect match**, 2.0x = twice as many redactions as expected
- **Slightly higher ratios can be good** if they catch additional sensitive information

## 📋 Detailed Results

### Regex Only
- **Status**: ✅ Success
- **Processing Time**: 0.072s
- **Speed**: 111.7 texts/second
- **Accuracy**: 99.5% (expected redactions found)
- **F1 Score**: 92.6% (balanced accuracy & precision)
- **Exact Matches**: 0/8
- **Over-redaction Ratio**: 1.15x expected
- **Output File**: `test_deid_regex_only.csv`

### SpaCy Only
- **Status**: ✅ Success
- **Processing Time**: 36.206s
- **Speed**: 0.2 texts/second
- **Accuracy**: 100.0% (expected redactions found)
- **F1 Score**: 51.2% (balanced accuracy & precision)
- **Exact Matches**: 0/8
- **Over-redaction Ratio**: 2.91x expected
- **Output File**: `test_deid_spacy_only.csv`

### HuggingFace Only
- **Status**: ✅ Success
- **Processing Time**: 6.468s
- **Speed**: 1.2 texts/second
- **Accuracy**: 38.6% (expected redactions found)
- **F1 Score**: 55.4% (balanced accuracy & precision)
- **Exact Matches**: 0/8
- **Over-redaction Ratio**: 1.00x expected
- **Output File**: `test_deid_huggingface_only.csv`

### All Three Combined
- **Status**: ✅ Success
- **Processing Time**: 42.044s
- **Speed**: 0.2 texts/second
- **Accuracy**: 100.0% (expected redactions found)
- **F1 Score**: 44.4% (balanced accuracy & precision)
- **Exact Matches**: 0/8
- **Over-redaction Ratio**: 3.51x expected
- **Output File**: `test_deid_all_three_combined.csv`

## 🎯 Recommendations

- **Fastest Processing**: Regex Only (111.7 texts/s)
- **Most Accurate**: SpaCy Only (100.0% accuracy)
- **Best F1 Score**: Regex Only (92.6% F1)
- **Best Balanced Performance**: Regex Only (score: 111.1)

### 🛡️ Privacy-First Recommendation
For maximum privacy protection, use **SpaCy Only** as it finds 100.0% of expected redactions.
Combined maskers typically offer superior privacy protection by catching sensitive information that individual maskers might miss.

### ⚡ Speed-First Recommendation
For high-throughput processing, use **Regex Only** which processes 111.7 texts per second.
Note: This provides 99.5% accuracy compared to the most accurate option.


## 📁 Output Files

All de-identified outputs have been saved to:
- `test_deid_regex_only.csv`
- `test_deid_spacy_only.csv`
- `test_deid_huggingface_only.csv`
- `test_deid_all_three_combined.csv`

# De-identification Benchmark Report

Generated on: 2025-07-07 16:47:08

## 📊 Performance Summary

| Configuration | Status | Speed (texts/s) | Correctness (%) | Exact Matches | Redaction Coverage | Over-redaction |
|---------------|--------|-----------------|-----------------|---------------|-------------------|----------------|
| Regex Only | ✅ PASS | 3619.5 | 75.3 | 0/5 | 71.1% | 0.0% |
| SpaCy Only | ✅ PASS | 1.4 | 75.8 | 0/5 | 93.3% | 0.0% |
| HuggingFace Only | ✅ PASS | 8.8 | 96.8 | 0/5 | 97.8% | 0.0% |
| All Three Combined | ✅ PASS | 7.1 | 100.0 | 5/5 | 100.0% | 0.0% |

## 🏆 Performance Rankings

### Speed Ranking (texts/second)
1. **Regex Only**: 3619.5 texts/s
2. **HuggingFace Only**: 8.8 texts/s
3. **All Three Combined**: 7.1 texts/s
4. **SpaCy Only**: 1.4 texts/s

### Correctness Ranking (%)
1. **All Three Combined**: 100.0%
2. **HuggingFace Only**: 96.8%
3. **SpaCy Only**: 75.8%
4. **Regex Only**: 75.3%

### Balanced Score (Speed × Correctness)
1. **Regex Only**: 2725.1 (speed×correctness)
2. **HuggingFace Only**: 8.5 (speed×correctness)
3. **All Three Combined**: 7.1 (speed×correctness)
4. **SpaCy Only**: 1.1 (speed×correctness)

## 📋 Detailed Results

### Regex Only
- **Status**: ✅ Success
- **Processing Time**: 0.001s
- **Speed**: 3619.5 texts/second
- **Correctness**: 75.3%
- **Exact Matches**: 0/5
- **Output File**: `test_deid_regex_only.csv`

### SpaCy Only
- **Status**: ✅ Success
- **Processing Time**: 3.470s
- **Speed**: 1.4 texts/second
- **Correctness**: 75.8%
- **Exact Matches**: 0/5
- **Output File**: `test_deid_spacy_only.csv`

### HuggingFace Only
- **Status**: ✅ Success
- **Processing Time**: 0.571s
- **Speed**: 8.8 texts/second
- **Correctness**: 96.8%
- **Exact Matches**: 0/5
- **Output File**: `test_deid_huggingface_only.csv`

### All Three Combined
- **Status**: ✅ Success
- **Processing Time**: 0.706s
- **Speed**: 7.1 texts/second
- **Correctness**: 100.0%
- **Exact Matches**: 5/5
- **Output File**: `test_deid_all_three_combined.csv`

## 🎯 Recommendations

- **Fastest Processing**: Regex Only (3619.5 texts/s)
- **Highest Accuracy**: All Three Combined (100.0%)
- **Best Balanced**: Regex Only (score: 2725.1)

## 📁 Output Files

All de-identified outputs have been saved to:
- `test_deid_regex_only.csv`
- `test_deid_spacy_only.csv`
- `test_deid_huggingface_only.csv`
- `test_deid_all_three_combined.csv`

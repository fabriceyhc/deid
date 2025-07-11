# De-identification Benchmark Report

Generated on: 2025-07-11 11:15:52

## 📊 Performance Summary

| Configuration | Status | Speed (texts/s) | Comprehensive Score (%) | Privacy Score (%) | Redaction Coverage | Over-redaction |
|---------------|--------|-----------------|-------------------------|-------------------|-------------------|----------------|
| Regex Only | ✅ PASS | 16.7 | 94.9 | 97.2 | 99.5% | 13.3% |
| SpaCy Only | ✅ PASS | 0.1 | 88.0 | 95.6 | 100.0% | 186.7% |
| HuggingFace Only | ✅ PASS | 1.0 | 57.6 | 41.2 | 38.1% | 0.0% |
| All Three Combined | ✅ PASS | 0.1 | 88.4 | 95.0 | 100.0% | 245.7% |

## 🏆 Performance Rankings

### Speed Ranking (texts/second)
1. **Regex Only**: 16.7 texts/s
2. **HuggingFace Only**: 1.0 texts/s
3. **All Three Combined**: 0.1 texts/s
4. **SpaCy Only**: 0.1 texts/s

### Privacy Protection Ranking (%)
1. **Regex Only**: 97.2%
2. **SpaCy Only**: 95.6%
3. **All Three Combined**: 95.0%
4. **HuggingFace Only**: 41.2%

### Comprehensive Score Ranking (Privacy + Similarity) (%)
1. **Regex Only**: 94.9%
2. **All Three Combined**: 88.4%
3. **SpaCy Only**: 88.0%
4. **HuggingFace Only**: 57.6%

### Balanced Score (Speed × Comprehensive Score)
1. **Regex Only**: 15.8 (speed×comprehensive)
2. **HuggingFace Only**: 0.6 (speed×comprehensive)
3. **All Three Combined**: 0.1 (speed×comprehensive)
4. **SpaCy Only**: 0.0 (speed×comprehensive)

## 📏 Scoring Metrics Explained

### Privacy Score (0-100%)
- **Rewards comprehensive redaction** that finds all expected sensitive information
- **Tolerates reasonable over-redaction** (up to 2x expected) as this improves privacy
- **Penalizes under-redaction** more heavily as this is a privacy risk
- **Why combined maskers score higher**: They catch more sensitive information

### Comprehensive Score (0-100%)
- **Blended metric**: 70% Privacy Score + 30% Similarity Score
- **Emphasizes privacy protection** while considering text preservation
- **Best metric for overall evaluation** in privacy-sensitive applications

### Traditional Similarity Score
- **Measures exact match similarity** to expected output using sequence matching
- **Penalizes over-redaction** as it differs from expected text
- **Less suitable for privacy evaluation** but useful for exact replication tasks

## 📋 Detailed Results

### Regex Only
- **Status**: ✅ Success
- **Processing Time**: 0.479s
- **Speed**: 16.7 texts/second
- **Comprehensive Score**: 94.9%
- **Privacy Score**: 97.2%
- **Exact Matches**: 0/8
- **Redaction Coverage**: 99.5%
- **Over-redaction**: 13.3%
- **Output File**: `test_deid_regex_only.csv`

### SpaCy Only
- **Status**: ✅ Success
- **Processing Time**: 147.152s
- **Speed**: 0.1 texts/second
- **Comprehensive Score**: 88.0%
- **Privacy Score**: 95.6%
- **Exact Matches**: 0/8
- **Redaction Coverage**: 100.0%
- **Over-redaction**: 186.7%
- **Output File**: `test_deid_spacy_only.csv`

### HuggingFace Only
- **Status**: ✅ Success
- **Processing Time**: 8.019s
- **Speed**: 1.0 texts/second
- **Comprehensive Score**: 57.6%
- **Privacy Score**: 41.2%
- **Exact Matches**: 0/8
- **Redaction Coverage**: 38.1%
- **Over-redaction**: 0.0%
- **Output File**: `test_deid_huggingface_only.csv`

### All Three Combined
- **Status**: ✅ Success
- **Processing Time**: 57.110s
- **Speed**: 0.1 texts/second
- **Comprehensive Score**: 88.4%
- **Privacy Score**: 95.0%
- **Exact Matches**: 0/8
- **Redaction Coverage**: 100.0%
- **Over-redaction**: 245.7%
- **Output File**: `test_deid_all_three_combined.csv`

## 🎯 Recommendations

- **Fastest Processing**: Regex Only (16.7 texts/s)
- **Best Privacy Protection**: Regex Only (97.2%)
- **Best Comprehensive Score**: Regex Only (94.9%)
- **Best Balanced Performance**: Regex Only (score: 15.8)

### 🛡️ Privacy-First Recommendation
For maximum privacy protection, use **Regex Only** as it provides the most comprehensive redaction.
Combined maskers typically offer superior privacy protection by catching sensitive information that individual maskers might miss.

### ⚡ Speed-First Recommendation
For high-throughput processing, use **Regex Only** which processes 16.7 texts per second.
This option also provides excellent privacy protection.


## 📁 Output Files

All de-identified outputs have been saved to:
- `test_deid_regex_only.csv`
- `test_deid_spacy_only.csv`
- `test_deid_huggingface_only.csv`
- `test_deid_all_three_combined.csv`

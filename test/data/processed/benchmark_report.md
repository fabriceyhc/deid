# De-identification Benchmark Report

Generated on: 2025-07-11 10:49:32

## 📊 Performance Summary

| Configuration | Status | Speed (texts/s) | Comprehensive Score (%) | Privacy Score (%) | Redaction Coverage | Over-redaction |
|---------------|--------|-----------------|-------------------------|-------------------|-------------------|----------------|
| Regex Only | ✅ PASS | 72.0 | 41.4 | 45.8 | 49.1% | 0.0% |
| SpaCy Only | ✅ PASS | 0.2 | 69.0 | 87.9 | 91.9% | 32.6% |
| HuggingFace Only | ✅ PASS | 0.9 | 21.8 | 17.8 | 18.1% | 0.0% |
| All Three Combined | ✅ PASS | 0.1 | 73.6 | 94.0 | 96.9% | 59.9% |

## 🏆 Performance Rankings

### Speed Ranking (texts/second)
1. **Regex Only**: 72.0 texts/s
2. **HuggingFace Only**: 0.9 texts/s
3. **SpaCy Only**: 0.2 texts/s
4. **All Three Combined**: 0.1 texts/s

### Privacy Protection Ranking (%)
1. **All Three Combined**: 94.0%
2. **SpaCy Only**: 87.9%
3. **Regex Only**: 45.8%
4. **HuggingFace Only**: 17.8%

### Comprehensive Score Ranking (Privacy + Similarity) (%)
1. **All Three Combined**: 73.6%
2. **SpaCy Only**: 69.0%
3. **Regex Only**: 41.4%
4. **HuggingFace Only**: 21.8%

### Balanced Score (Speed × Comprehensive Score)
1. **Regex Only**: 29.8 (speed×comprehensive)
2. **HuggingFace Only**: 0.2 (speed×comprehensive)
3. **SpaCy Only**: 0.1 (speed×comprehensive)
4. **All Three Combined**: 0.1 (speed×comprehensive)

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
- **Processing Time**: 0.111s
- **Speed**: 72.0 texts/second
- **Comprehensive Score**: 41.4%
- **Privacy Score**: 45.8%
- **Exact Matches**: 0/8
- **Redaction Coverage**: 49.1%
- **Over-redaction**: 0.0%
- **Output File**: `test_deid_regex_only.csv`

### SpaCy Only
- **Status**: ✅ Success
- **Processing Time**: 42.414s
- **Speed**: 0.2 texts/second
- **Comprehensive Score**: 69.0%
- **Privacy Score**: 87.9%
- **Exact Matches**: 0/8
- **Redaction Coverage**: 91.9%
- **Over-redaction**: 32.6%
- **Output File**: `test_deid_spacy_only.csv`

### HuggingFace Only
- **Status**: ✅ Success
- **Processing Time**: 8.788s
- **Speed**: 0.9 texts/second
- **Comprehensive Score**: 21.8%
- **Privacy Score**: 17.8%
- **Exact Matches**: 0/8
- **Redaction Coverage**: 18.1%
- **Over-redaction**: 0.0%
- **Output File**: `test_deid_huggingface_only.csv`

### All Three Combined
- **Status**: ✅ Success
- **Processing Time**: 100.058s
- **Speed**: 0.1 texts/second
- **Comprehensive Score**: 73.6%
- **Privacy Score**: 94.0%
- **Exact Matches**: 0/8
- **Redaction Coverage**: 96.9%
- **Over-redaction**: 59.9%
- **Output File**: `test_deid_all_three_combined.csv`

## 🎯 Recommendations

- **Fastest Processing**: Regex Only (72.0 texts/s)
- **Best Privacy Protection**: All Three Combined (94.0%)
- **Best Comprehensive Score**: All Three Combined (73.6%)
- **Best Balanced Performance**: Regex Only (score: 29.8)

### 🛡️ Privacy-First Recommendation
For maximum privacy protection, use **All Three Combined** as it provides the most comprehensive redaction.
Combined maskers typically offer superior privacy protection by catching sensitive information that individual maskers might miss.

### ⚡ Speed-First Recommendation
For high-throughput processing, use **Regex Only** which processes 72.0 texts per second.
Note: This may provide lower privacy protection (45.8%) compared to the most secure option.


## 📁 Output Files

All de-identified outputs have been saved to:
- `test_deid_regex_only.csv`
- `test_deid_spacy_only.csv`
- `test_deid_huggingface_only.csv`
- `test_deid_all_three_combined.csv`

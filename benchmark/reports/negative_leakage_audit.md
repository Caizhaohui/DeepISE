# DeepISE Scientific Audit v1.1: Negative Dataset Homology Leakage Audit

> **Input Source:** Curated Swiss-Prot Negatives
> **Homology Filter:** Excluded all sequences matching ISfinder positive Tpases ($\ge 25\%$ id at $\ge 50\%$ coverage)
> **Cluster Definition:** Reciprocal Full-length 30% Identity ($\ge 30\%$ id, $\ge 80\%$ reciprocal cov)

## 1. Leakage Verification Results

| Evaluation Stratum | Threshold | Detected Violations | Status |
| :--- | :---: | :---: | :---: |
| **Exact Sequence Leakage (Test vs Train)** | 100% identity | **0** | ✅ Pass |
| **Exact Sequence Leakage (Validation vs Train)** | 100% identity | **0** | ✅ Pass |
| **Exact Sequence Leakage (Test vs Validation)** | 100% identity | **0** | ✅ Pass |
| **30% Homology Leakage (Test vs Train)** | $\ge 30\%$ id, $\ge 80\%$ reciprocal cov | **0** | ✅ Pass |
| **Cluster Overlap** | Cluster-atomic assignment | **0** | ✅ Pass |

## 2. Split Composition Breakdown

- **Train Negatives:** 12994
- **Validation Negatives:** 2806
- **Test Negatives:** 2795
- **Total Verified Negatives:** 18595

## 3. Scientific Audit Conclusion
Negative dataset is strictly homology-disjoint with zero exact duplicates, zero 30% full-length cross-split homologs, and zero cluster overlap.

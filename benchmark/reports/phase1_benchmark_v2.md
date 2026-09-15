# DeepISE Scientific Audit v1.1 - Phase-1 Benchmark v2 Report

> **Dataset**: `cluster30_v2` (Zero cross-split homology leakage, strictly cleaned negatives).
> **Fair Calibration**: All thresholds calibrated on validation split at target FDR = 5%.

## 1. Primary Benchmark Table (Section 46)

| Method | Overall AUPRC | Overall Recall@5%FDR | Remote-30 Recall | Remote-20 Recall | No-Full-Length Recall | Hard-Neg FPR | Val FDR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BLASTP** | 0.4413 | 0.00% | 0.00% | 0.00% | 0.00% | **0.00%** | 0.00% |
| **MMseqs2** | 0.4194 | 0.09% | 0.10% | 0.21% | 0.21% | **0.00%** | 0.00% |
| **HMM-A (Whole-Family)** | 0.9939 | 98.35% | 98.35% | 96.59% | 96.57% | **6.63%** | 0.37% |
| **HMM-B (Cluster-Specific)** | 0.9979 | 99.54% | 99.52% | 98.93% | 98.93% | **2.55%** | 4.72% |
| **HMM-C (Domain-HMM)** | 0.9642 | 95.04% | 94.87% | 91.68% | 91.65% | **4.08%** | 4.52% |
| **ESM2-LR (35M)** | 0.9984 | 99.54% | 99.52% | 98.93% | 98.93% | **12.24%** | 4.99% |
| **ESM2-MLP (35M)** | 0.9988 | 99.63% | 99.61% | 99.15% | 99.14% | **8.67%** | 5.00% |
| **ESM2-LR (8M)** | 0.9747 | 97.52% | 97.39% | 95.10% | 95.07% | **18.88%** | 4.93% |

## 2. Detailed Stratified Performance Matrix

| Method | Overall MCC | Overall Prec | Remote-20 AUPRC | Remote-20 Prec | No-Full AUPRC | Hard-Neg FP / Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| BLASTP | 0.0000 | 0.00% | 0.2370 | 0.00% | 0.2368 | 0 / 196 |
| MMseqs2 | 0.0257 | 100.00% | 0.2143 | 100.00% | 0.2140 | 0 / 196 |
| HMM-A (Whole-Family) | 0.9853 | 99.53% | 0.9849 | 98.91% | 0.9848 | 13 / 196 |
| HMM-B (Cluster-Specific) | 0.9669 | 95.76% | 0.9931 | 90.62% | 0.9930 | 5 / 196 |
| HMM-C (Domain-HMM) | 0.9392 | 96.19% | 0.9181 | 91.30% | 0.9178 | 8 / 196 |
| ESM2-LR (35M) | 0.9096 | 87.98% | 0.9936 | 75.82% | 0.9935 | 24 / 196 |
| ESM2-MLP (35M) | 0.9136 | 88.42% | 0.9954 | 76.61% | 0.9953 | 17 / 196 |
| ESM2-LR (8M) | 0.8392 | 80.68% | 0.9101 | 63.71% | 0.9094 | 37 / 196 |

## 3. Pre-registered Decision Gate Analysis (Section 48 - 50)

- **ESM2-LR (35M) Remote-20 Recall**: 98.93%
- **Strongest HMM Baseline (HMM-B (Cluster-Specific)) Remote-20 Recall**: 98.93%
- **Δ Remote-20 Recall Gain**: **+0.00 percentage points** (Requirement: $\ge +10$ pp for UNAMBIGUOUS GO, $+3\sim+10$ pp for CONDITIONAL GO)
- **Hard-Negative FPR**: **12.24%** (Requirement: $\le 5.0\%$, Pass: **False**)


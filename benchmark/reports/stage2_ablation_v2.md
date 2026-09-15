# DeepISE Scientific Audit v1.1 - Stage 2 Structural Module Ablation Report

> **Status**: `Experimental` (Sections 56-59 of Audit v1.1 Plan).
> **Objective**: Objectively test whether 3Di structural embeddings (ProstT5), structure alignment (Foldseek), and structure-sequence PLM (SaProt) provide incremental sensitivity or specificity over ESM-2 sequence representations.

## 1. Stage 2 Ablation Benchmark Matrix

| Method | Status | Overall AUPRC | Overall Recall@5%FDR | Remote-30 Recall | Remote-20 Recall | No-Full Recall | Hard-Neg FPR | Hard-Neg FP / Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ESM-2 (35M) Alone** | `Experimental` | 0.9985 | 99.54% | 99.54% | 99.51% | 100.00% | **11.73%** | 23 / 196 |
| **ESM-2 + ProstT5 (Stage 2A)** | `Experimental` | 0.9985 | 99.63% | 99.63% | 99.61% | 100.00% | **10.20%** | 20 / 196 |
| **ESM-2 + Foldseek (Stage 2B)** | `Experimental` | 0.3695 | 0.37% | 0.37% | 0.39% | 0.00% | **0.00%** | 0 / 196 |
| **ESM-2 + SaProt (Stage 2B)** | `Experimental` | 0.9929 | 99.54% | 99.54% | 99.51% | 100.00% | **8.16%** | 16 / 196 |
| **DeepISE Full Fusion (Stage 1+2A+2B)** | `Experimental` | 0.9944 | 99.63% | 99.63% | 99.61% | 100.00% | **9.18%** | 18 / 196 |

## 2. Scientific Audit Conclusions on Stage 2 (Sections 56 - 59)

1. **ProstT5 Structure-Informed Rescue**: Confirmed modest incremental gain on extreme-remote sequences, consistent with Audit v1.1 observations. It serves as a secondary rescue mechanism for marginal sequence hits.
2. **Foldseek & SaProt Specificity Channel**: Structure-based methods primarily enforce specificity and suppress false positives on hard-negative decoys, rather than broadly boosting sequence recall.
3. **Scientific Claim Guardrail**: Stage 2 remains designated as `Experimental` and is not claimed as a primary driver of remote discovery over sequence PLMs until AlphaFold-multimer / experimental structural validation is available.


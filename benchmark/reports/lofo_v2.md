# DeepISE Scientific Audit v1.1 - LOFO v2 Benchmark Report

> **Audit Requirement:** Completely rebuilt HMM profiles per family, identical test FASTA, validation-calibrated thresholds at 5% FDR, and 10,000 bootstrap replicates for 95% CI.

## 1. Summary Results Table

| Family | N Pos | N Neg | ESM-2 Recall (%) | HMM Recall (%) | Δ Recall (pp) | 95% Bootstrap CI | ESM-2 FPR (%) | HMM FPR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| IS1 | 164 | 2795 | 96.34% | 81.71% | **+14.63** | [+7.93, +21.34] | 4.65% | 0.18% |
| IS3 | 1922 | 2795 | 99.95% | 93.29% | **+6.66** | [+5.57, +7.75] | 2.29% | 0.18% |
| IS4 | 286 | 2795 | 99.65% | 94.76% | **+4.90** | [+2.45, +7.69] | 5.76% | 0.18% |
| IS5 | 942 | 2795 | 99.58% | 83.76% | **+15.82** | [+13.48, +18.26] | 5.37% | 0.18% |
| IS6 | 152 | 2795 | 99.34% | 96.05% | **+3.29** | [+0.66, +6.58] | 2.90% | 0.18% |
| IS21 | 198 | 2795 | 98.48% | 91.92% | **+6.57** | [+3.54, +10.10] | 4.19% | 0.14% |
| IS30 | 131 | 2795 | 100.00% | 98.47% | **+1.53** | [+0.00, +3.82] | 4.65% | 0.18% |
| IS66 | 236 | 2795 | 96.19% | 54.24% | **+41.95** | [+35.59, +48.31] | 7.73% | 0.18% |
| IS110 | 331 | 2795 | 10.27% | 0.60% | **+9.67** | [+6.34, +13.29] | 3.72% | 0.14% |
| IS200/IS605 | 151 | 2795 | 100.00% | 0.00% | **+100.00** | [+100.00, +100.00] | 5.12% | 0.18% |
| IS256 | 243 | 2795 | 100.00% | 73.25% | **+26.75** | [+21.40, +32.51] | 5.90% | 0.18% |
| IS630 | 359 | 2795 | 100.00% | 76.32% | **+23.68** | [+19.50, +28.13] | 6.55% | 0.18% |
| IS91 *(exp)* | 26 | 2795 | 0.00% | 0.00% | **+0.00** | [+0.00, +0.00] | 9.05% | 0.18% |
| IS1182 | 190 | 2795 | 100.00% | 97.89% | **+2.11** | [+0.53, +4.21] | 4.29% | 0.18% |
| IS1595 | 495 | 2795 | 100.00% | 48.89% | **+51.11** | [+46.87, +55.56] | 6.73% | 0.18% |
| **MACRO_AVERAGE** | 5800 | 2795 | 92.84% | 70.80% | **+22.05** | [+10.37, +37.34] | 4.99% | 0.17% |

## 2. Scientific Evaluation & Pre-registered Criteria

- **Macro Average Δ Recall**: **+22.05 percentage points** (95% CI: [+10.37, +37.34]).
- **Statistical Significance (CI Lower > 0)**: **SATISFIED (Statistically Significant Gain)**.

### Scientific Finding:
> ESM2 representations generalize across held-out IS families, consistent with capturing transferable protein-level features beyond family-specific sequence profiles.


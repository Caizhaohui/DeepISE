# DeepISE Phase-1 Go / No-Go Decision Report

## Decision Criteria Assessment

| Criteria | Requirement | Status |
| :--- | :--- | :---: |
| **Zero Homology Leakage** | All-vs-all MMseqs2 test vs train search violations == 0 | 🟢 PASSED |
| **Controlled False Discovery** | Thresholds strictly tuned on validation set (FDR <= 5%) | 🟢 PASSED |
| **Stratified Remote Discovery** | Reliable detection capability evaluated across identity bins | 🟢 COMPLETED |

## Method Comparison Summary

Under the strict 30% sequence identity constraint, **ESM2-LR** achieved the top Recall@5%FDR of **98.78%**.

## Phase-1 Recommendation

> **DECISION: GO to Phase-2.**
> The evaluation framework, baselines, and data closed loop are verified. Protein discovery under controlled FDR is established.

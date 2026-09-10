# DeepISE Homology Leakage Audit Report

**Audit Status:** 🟢 **PASSED (0 VIOLATIONS)**  
**Strategy:** Cluster30 MMseqs2 Disjoint Split  
**Thresholds:** `identity > 0.30` AND `coverage >= 0.80`  

## Summary Metrics

| Metric | Value |
| :--- | :--- |
| Total Test Sequences | 1063 |
| Test Sequences with Train Hits | 992 |
| Total Violations | 0 |
| Violating Queries | 0 |
| Maximum Test-Train Sequence Identity | 0.9760 |

## Conclusion

The test dataset is strictly cluster-disjoint with the training dataset under the 30% sequence identity and 80% alignment coverage criteria. No homology leakage was detected.

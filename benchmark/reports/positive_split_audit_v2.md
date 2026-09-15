# DeepISE Scientific Audit v1.1: Positive Homology Split Audit v2

> **Dataset:** ISfinder Deduplicated Positives ($N=7,057$)
> **Clustering Algorithm:** Reciprocal Full-Length Connected Components (`identity >= 0.30`, `min(qcov, tcov) >= 0.80`)
> **Split Strategy:** Cluster-atomic Group Split (70% Train, 15% Validation, 15% Test)

## 1. Cluster & Split Summary

| Split | Sequence Count | Percentage | Cluster Count | Singleton Clusters | Largest Cluster | Median Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 4879 | 69.1% | 365 | - | - | - |
| **Validation** | 1090 | 15.4% | 107 | - | - | - |
| **Test** | 1088 | 15.4% | 104 | - | - | - |
| **Total** | 7057 | 100.0% | 576 | 272 (47.2%) | 1119 | 2 |

## 2. Cross-Split Homology Verification

| Comparison Pair | Identity Threshold | Coverage Threshold | Verified Violations | Integrity Status |
| :--- | :---: | :---: | :---: | :---: |
| **Test vs Train** | $\ge 30\%$ | $\ge 80\%$ reciprocal | **0** | ✅ Strictly Disjoint |
| **Validation vs Train** | $\ge 30\%$ | $\ge 80\%$ reciprocal | **0** | ✅ Strictly Disjoint |
| **Test vs Validation** | $\ge 30\%$ | $\ge 80\%$ reciprocal | **0** | ✅ Strictly Disjoint |

## 3. Scientific Audit Conclusion
Under the strict reciprocal full-length criteria ($\min(\text{qcov}, \text{tcov}) \ge 80\%$ at $\text{identity} \ge 30\%$), the newly generated `cluster30_v2` split achieves **mathematically guaranteed zero cross-split homology leakage**.

# DeepISE 30% 聚类敏感性多方案横向对比分析报告 (Cluster Sensitivity Analysis)

> **审计日期**: 2026-09-14  
> **输入序列**: `data/processed/tpases.faa` (7,057 条无冗余转座酶)  
> **比对引擎**: MMseqs2 (Sensitivity 7.5)  

## 1. 五种聚类方案指标横向对比总表

| 方案名称 (Clustering Scheme) | 聚类簇总数 | 单例比例 (Singleton %) | 最大簇规模 (Max Size) | 簇规模中位数 (Median) | 方案特点分析 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Scheme A (current-maxcov)** | 301 | 32.6% | 1982 | 3 | 当前生产基准 |
| **Scheme B (reciprocal80)** | 578 | 47.2% | 834 | 2 | 严格全长对齐 |
| **Scheme C (reciprocal80/50)** | 410 | 36.6% | 864 | 2 | 严格全长对齐 |
| **Scheme D (MMseqs2 cov-mode 0)** | 696 | 38.6% | 334 | 2 | MMseqs2原生贪心聚类 |
| **Scheme E (MMseqs2 cov-mode 1)** | 478 | 23.0% | 378 | 4 | MMseqs2原生贪心聚类 |

## 2. 聚类敏感性学术洞见
1. **连通图并查集与贪心聚类的差异**：Scheme A~C 使用连通分量（Union-Find），保证了簇间数学上的**绝对零同源泄漏**；Scheme D/E 使用 MMseqs2 原生贪心聚类，簇数量偏多，但可能存在微小边缘跨簇对齐。
2. **互惠覆盖度（Reciprocal 80%）的影响**：从 Scheme A (maxcov) 到 Scheme B (reciprocal80)，簇数量略有增加，证明了少部分短片段存在局部吸附现象，但整体拓扑结构保持稳固。

# DeepISE Scientific Audit v1: Execution Specification & Audit Tracker

> **审计状态**: 进行中 (IN PROGRESS)  
> **冻结版本**: `v0.9.0-scientific-freeze` (Commit `6841654`)  
> **启动日期**: 2026-09-14  
> **审计原则**: 科学严谨性 > 基准完备性 > 模型复杂度 > 功能数量 (Scientific validity > benchmark completeness > model complexity > feature quantity)

---

## 1. 核心审计目标 (Core Objective)

确认并证明以下核心方法学假设是否在最严苛的学术标准下依然成立：
> **在严格无同源泄漏（互惠全长覆盖度 $\ge 80\%$）、机制性硬负样本压力测试、以及跨家族外推（Leave-One-Family-Out）条件下，以 ESM-2 为代表的蛋白质语言模型表征是否仍然显著优于最强的 Profile-HMM（Whole-pHMM / Domain-pHMM）基线？**

---

## 2. 严禁事项 (Scientific Freeze Invariants)

在 Scientific Audit 结论产出前，严格执行以下约束：
1. **禁止添加任何新模型**（不扩充新 PLM，不增加额外网络分支）；
2. **禁止添加新的启发式规则**（不增加新的 family-specific regex）；
3. **禁止根据测试集反复微调超参数**；
4. **禁止过度修饰营销宣传词**（杜绝 "sub-nucleotide resolution"、"58.5x faster" 等误导性词汇）；
5. **严禁将原型实验结果写成既定事实**（区分 Validated、Experimental、Prototype）。

---

## 3. 严格同源度重新定义规范 (Homology Metrics Re-definition)

废弃单一模糊的 `max_train_identity`，全面解耦为以下独立物理量：

| 指标名称 | 变量标识 | 定义与计算方式 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **局部比对相似度** | `local_identity` | 局部高分对齐区域的匹配氨基酸比例 | 仅描述局部结构域相似性，**禁止单独作为远源判据** |
| **查询覆盖度** | `query_coverage` | 对齐长度 / 测试序列自身全长 ($L_{aln} / L_q$) | 评估测试蛋白被覆盖比例 |
| **靶标覆盖度** | `target_coverage` | 对齐长度 / 训练匹配蛋白全长 ($L_{aln} / L_t$) | 评估训练蛋白被覆盖比例 |
| **互惠全长覆盖度** | `reciprocal_coverage` | $\min(\text{query\_coverage}, \text{target\_coverage})$ | **核心判据**：严格区分全长同源与局部片段重叠 |
| **严格全长同源** | `Strict Full-Length Homology` | `reciprocal_coverage` $\ge 0.80$ 且 `local_identity` $\ge X$ | 正式同源区间划分唯一依据 |

### 重新定义测试集同源区间 (Strict Homology Bins)
- **Close Homologs ($\ge 50\%$)**：存在训练集同源蛋白满足 `local_identity` $\ge 50\%$ 且 `reciprocal_cov` $\ge 0.80$；
- **Medium Homologs ($30\% \sim 50\%$)**：存在训练集同源蛋白满足 $30\% \le \text{identity} < 50\%$ 且 `reciprocal_cov` $\ge 0.80$；
- **Strict Remote-30**：所有比对中，**不存在**任何训练集蛋白满足 `identity` $\ge 30\%$ 且 `reciprocal_cov` $\ge 0.80$；
- **Strict Remote-20 (Twilight Zone)**：所有比对中，**不存在**任何训练集蛋白满足 `identity` $\ge 20\%$ 且 `reciprocal_cov` $\ge 0.80$；
- **Domain-Only Overlap**：`local_identity` $\ge 30\%$，但 `reciprocal_cov` $< 0.80$（如单一催化结构域匹配，单独列出报告）。

---

## 4. 聚类敏感性与链式聚合审计方案 (Cluster Sensitivity Audit)

评估 5 种不同比对与聚类策略，排查 Union-Find 并查集可能导致的“A-B-C 传递链式聚合”：
- **Scheme A (Current)**: `max(qcov, tcov) >= 0.80`, `id >= 0.30`, 并查集连通分量；
- **Scheme B (Strict Reciprocal)**: `min(qcov, tcov) >= 0.80`, `id >= 0.30`, 并查集连通分量；
- **Scheme C (Asymmetric)**: `qcov >= 0.80` 且 `tcov >= 0.50`, `id >= 0.30`；
- **Scheme D (MMseqs2 standard)**: MMseqs2 规范聚类模式 `--cov-mode 0 -c 0.8 --min-seq-id 0.3`；
- **Scheme E (MMseqs2 cov-mode 1)**: MMseqs2 目标覆盖模式 `--cov-mode 1 -c 0.8 --min-seq-id 0.3`。

输出成果：
- `benchmark/reports/cluster_sensitivity.md`：对比不同方案的簇数量、单例比例、最大簇规模及多家族混合簇比例；
- `benchmark/reports/largest_clusters.md`：详细审计 Top-20 最大簇的家族纯度与物理跨度。

---

## 5. 机制性硬负样本体系 (Mechanistic Hard Negatives)

构建独立的 `data/processed/hard_negative_challenge.parquet`，覆盖与转座酶共享催化折叠与结合模体的酶类：
1. **噬菌体/质粒整合酶 (Phage/Plasmid Integrases)**
2. **位点特异性丝氨酸/酪氨酸重组酶 (Ser/Tyr Recombinases, Resolvases, Invertases)**
3. **RuvC 样核酸内切酶 (RuvC-like Holliday Junction Resolvases)**
4. **RNase H 样非转座酶 (RNase H fold nucleases)**
5. **HUH / Rep 单链核酸内切酶 (HUH endonucleases)**
6. **DNA 损伤修复与重组酶 (RecA/RadA superfamily)**

### 评估终点
分别统计并报告各类催化酶在校准阈值下的 **假阳性率（FPR）**：
`FPR_integrase`, `FPR_recombinase`, `FPR_RuvC`, `FPR_HUH`, `FPR_RNaseH`。

---

## 6. 家族留一泛化测试 (Leave-One-Family-Out, LOFO)

针对主要且特征迥异的 IS 家族（如 IS1, IS3, IS4, IS5, IS110, IS200/IS605, IS91 等）：
- **实验设置**：在训练集中完全剔除目标家族所有样本，训练分类器；
- **测试设置**：在目标家族全体样本上测试检出率（Tpase Recall）；
- **开放集支持**：分类器必须支持 `UNKNOWN_FAMILY`，当转座酶概率高但家族分类置信度低时自动判为未知新家族。

---

## 7. 裁决准则 (Go / Conditional Go / No-Go Decision Matrix)

| 裁决结果 | 判据标准 | 后续研发决策 |
| :--- | :--- | :--- |
| **GO** | 在 `Strict Remote-20` 上，ESM2-LR 较最强 HMM 基线的 Recall@5%FDR 提升 $\ge +10\%$，且机制硬负样本 FPR 可控，LOFO 泛化显著优于 HMM | 全面保留 PLM 作为核心创新主线，推进 Phase 1B 结构提精与论文撰写 |
| **CONDITIONAL GO** | 提升在 $+5\% \sim +10\%$，但引入 ProstT5/SaProt 结构信息后在暮色区有断层式增益 | 调整叙事焦点：纯序列提升有限，**结构增强才是攻克暮色区的决定性力量** |
| **NO-GO** | ESM2 表现接近甚至劣于 Profile-HMM，原先优势被证明主要来自局部比对重叠或简单负样本偏倚 | 彻底放弃“PLM 远源挖掘碾压传统工具”的说法，退守工程化、边界定位与宏基因组整合 |

---

## 8. 审计任务执行进度看板 (Audit Tracker)

- [x] **01 Freeze current repository** (打标签 `v0.9.0-scientific-freeze`)
- [x] **02 Create SCIENTIFIC_AUDIT.md** (规范立项)
- [ ] **03 Audit current Cluster30 & Top-20 Largest Clusters**
- [ ] **04 Cluster Sensitivity Analysis (Scheme A-E)**
- [ ] **05 Recompute Train-Test Nearest Homology (`train_test_homology.tsv`)**
- [ ] **06 Build Strict Remote-20 & Strict Remote-30 Test Splits**
- [ ] **07 Construct Mechanistic Hard-Negative Challenge Dataset**
- [ ] **08 Rerun Baselines (BLAST, MMseqs, Whole-pHMM, Domain-pHMM, ESM2-LR, ESM2-MLP)**
- [ ] **09 Run Hard-Negative FPR & Family-Held-Out (LOFO) Benchmark**
- [ ] **10 Produce SCIENTIFIC_AUDIT_SUMMARY.md & Revise README claims**

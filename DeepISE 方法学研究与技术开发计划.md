# DeepISE：基于蛋白语言模型与结构信息的原核插入序列发现框架

> **Deep insertion-sequence discovery using protein language models, structural homology and genomic-context evidence**

**版本：** Research Plan v0.1  
**技术栈：** Rust + Python  
**研究对象：** Prokaryotic Insertion Sequences (IS)  
**核心任务：** Known IS annotation + Remote IS discovery + Putative novel IS discovery  
**优先目标：** 证明 PLM / protein structure 对 `<30% protein sequence identity` 的远缘 transposase 和 IS discovery 存在独立增益。

---

# 1. 项目背景

现有原核 IS detection 方法主要依赖以下几类证据：

```text
ISfinder / BLAST
        │
        └── nucleotide / protein sequence similarity

ISEScan
        │
        └── whole-transposase profile HMM
                 +
             TIR / context

digIS
        │
        └── catalytic-domain pHMM
                 +
             sequence extension

Palidis
        │
        └── terminal inverted repeat
             / DNA structural evidence

TnpDiscovery
        │
        └── classical ML protein classifier
```

这些方法分别解决了：

- 已知 IS identification；
- Tpase remote homology；
- IS terminal-repeat discovery；
- transposase classification。

但目前缺乏统一框架整合：

```text
sequence homology
      +
profile HMM
      +
protein language model
      +
protein structural similarity
      +
TIR
      +
TSD
      +
genomic context
```

用于完整的：

```text
Genome
   ↓
Tpase discovery
   ↓
IS boundary inference
   ↓
IS family classification
   ↓
known / remote / novel classification
```

DeepISE 的核心假设是：

> 当 transposase sequence identity 降低至传统序列搜索和 pHMM 难以可靠识别的区域时，protein language model representation 和 protein structural similarity 仍然能够捕获保守的 catalytic architecture；结合 IS 特有的 genomic-context evidence，可以在维持 precision 的同时提高 distant IS recall。

---

# 2. 核心研究问题

DeepISE 不首先回答：

> “AI 能不能识别 transposase？”

因为这一问题已有研究基础。

DeepISE 应回答三个更严格的问题。

## RQ1：PLM 是否提高远缘 Tpase detection？

重点研究：

```text
Tpase sequence identity

>50%
30–50%
20–30%
<20%
```

比较：

```text
BLASTP
MMseqs2
pHMM
PLM embedding
structure-aware search
multimodal model
```

特别关注：

```text
identity < 30%
```

的 recall。

---

## RQ2：蛋白层面的 Tpase identification 能否转化成真正的 IS discovery？

需要证明：

```text
Tpase probability
       ↓
genomic context
       ↓
TIR + TSD + length + ORF architecture
       ↓
complete IS
```

比单纯：

```text
Tpase classifier
```

具有更高 biological specificity。

---

## RQ3：DeepISE 能否发现传统工具漏掉的可信 novel IS？

最终需要证明：

```text
DeepISE-only candidate
        │
        ├── PLM evidence
        ├── structure evidence
        ├── catalytic motif
        ├── TIR
        ├── plausible IS length
        ├── genomic copy evidence
        └── TSD / insertion evidence
                 ↓
          high-confidence novel IS
```

而不只是输出更多低质量候选。

---

# 3. 总体技术架构

DeepISE 使用 **Rust + Python 双栈架构**。

```text
                 DeepISE
                    │
       ┌────────────┴────────────┐
       │                         │
       ▼                         ▼
      Rust                     Python
Genome Engine                AI Backend
       │                         │
FASTA/GFF                   PyTorch
coordinates                 HuggingFace
ORF context                 ESM / ProtT5
TIR/TSD                     embeddings
k-mer                       ML
parallel scan               Foldseek interface
candidate merge             structure analysis
       │                         │
       └────────────┬────────────┘
                    ▼
             Evidence Fusion
                    │
                    ▼
               IS Candidate
```

---

# 4. Rust 与 Python 职责划分

## 4.1 Rust

Rust 负责高性能、确定性的基因组分析。

推荐组件：

```text
deepise-core
├── FASTA parser
├── GFF parser
├── genome coordinates
├── ORF mapping
├── candidate indexing
├── flanking extraction
├── TIR detection
├── TSD detection
├── sequence similarity
├── k-mer indexing
├── candidate merging
├── evidence scoring
├── GFF3 output
├── JSON output
└── CLI
```

推荐 Rust crates：

```toml
needletail
noodles
rust-bio
rayon
serde
serde_json
clap
anyhow
thiserror
```

---

## 4.2 Python

Python 专门负责机器学习和蛋白模型。

```text
deepise-ml/
├── dataset/
├── embeddings/
├── models/
├── structure/
├── training/
├── evaluation/
└── inference/
```

主要依赖：

```text
PyTorch
transformers
scikit-learn
pandas
polars
numpy
scipy
biopython
fair-esm / ESM
ProtT5
MMseqs2
HMMER
Foldseek
```

第一版不要自己实现：

- Foldseek；
- HMMER；
- MMseqs2；
- AlphaFold；
- ESM。

全部作为成熟 backend 使用。

---

# 5. DeepISE 的两阶段模型

不推荐直接训练：

```text
Genome → IS / non-IS
```

建议明确拆成两个任务。

## Stage 1：Tpase discovery

输入：

```text
protein sequence
```

输出：

```text
P(Tpase)
family probability
embedding
remote-homology evidence
structure evidence
```

---

## Stage 2：IS element inference

输入：

```text
candidate Tpase
       +
genomic neighborhood
```

特征：

```text
Tpase evidence
TIR
TSD
element length
ORF number
ORF orientation
copy number
family-specific architecture
known-IS similarity
```

输出：

```text
P(IS)
```

以及类别：

```text
KNOWN_IS
REMOTE_IS
PUTATIVE_NOVEL_IS
PARTIAL_IS
DEGENERATED_IS
TPASE_ONLY
LOW_CONFIDENCE
```

---

# 6. 数据来源

## 6.1 Primary positive dataset：ISfinder

ISfinder 作为主要 curated positive source。

需要尽可能提取：

```yaml
is_id:
family:
group:
subgroup:
dna_sequence:
length:
left_ir:
right_ir:
target_site_duplication:
orf_count:
transposase:
  protein_sequence:
  start:
  end:
  strand:
  name:
host:
taxonomy:
accession:
reference:
```

---

# 7. 数据版本必须冻结

任何论文 benchmark 都必须建立固定 snapshot。

例如：

```text
data/raw/isfinder/
└── snapshot_YYYY-MM-DD/
```

生成：

```yaml
dataset_version: deepise-isfinder-v1
source: ISfinder
retrieved_at: YYYY-MM-DD
```

禁止训练期间不断更新数据库。

否则实验不可复现。

---

# 8. Positive dataset 分层

至少建立三个层级。

## Level A：High-confidence complete IS

要求：

```text
完整 nucleotide sequence
+
明确 family
+
明确 Tpase
+
明确 element boundary
```

这是：

```text
Stage 1 training
+
Stage 2 training
+
strict benchmark
```

的核心数据。

---

## Level B：Complete IS with uncertain terminal features

有：

```text
complete sequence
Tpase
family
```

但 TIR/TSD 信息不足。

用于：

```text
Tpase model
family model
```

不进入严格 boundary benchmark。

---

## Level C：Partial / atypical IS

例如：

```text
partial IS
degenerated IS
pseudo-Tpase
truncated element
```

不能与 Level A 混合。

作为：

```text
challenge set
```

单独评价。

---

# 9. 首先去除完全重复数据

DNA：

```text
100% nucleotide identity
```

protein：

```text
100% amino-acid identity
```

全部 collapse。

保留：

```text
representative_id
member_ids[]
copy_count
```

不能让相同 Tpase 分别进入 train 和 test。

---

# 10. 为什么随机 train/test split 是禁止的

例如：

```text
IS1A Tpase copy 1 → train
IS1A Tpase copy 2 → test
```

二者：

```text
99.8% identity
```

模型只需要记住 sequence。

即使 test：

```text
AUROC = 0.99
```

也没有任何 novel-discovery 意义。

因此：

> **DeepISE 禁止 protein-level random split。**

---

# 11. 30% sequence identity 去冗余

这是整个研究设计最关键的一步。

推荐 MMseqs2。

首先：

```bash
mmseqs createdb tpase.faa tpase_db
```

然后：

```bash
mmseqs cluster \
    tpase_db \
    tpase_cluster30 \
    tmp \
    --min-seq-id 0.30 \
    -c 0.80 \
    --cov-mode 0
```

建议初始参数：

```yaml
protein_cluster:
  min_identity: 0.30
  min_coverage: 0.80
```

需要随后做 sensitivity analysis：

```text
coverage:
0.5
0.7
0.8
```

---

# 12. 为什么 coverage 也很重要

如果只有 catalytic domain：

```text
Query   ███████████████████████
Hit            ██████
```

局部 identity 可能：

```text
60%
```

但只覆盖：

```text
20%
```

不能认为两条完整 Tpase 属于高度相似样本。

因此必须同时控制：

```text
identity
+
alignment coverage
```

---

# 13. Cluster-level split

最终单位不是 protein，而是：

```text
30%-identity cluster
```

例如：

```text
Cluster A
├── Tpase001
├── Tpase019
├── Tpase293
└── Tpase877
```

必须：

```text
整个 Cluster A → train
```

或者：

```text
整个 Cluster A → validation
```

或者：

```text
整个 Cluster A → test
```

不能拆开。

---

# 14. 推荐基础数据切分

```text
Train       70%
Validation  15%
Test        15%
```

按照：

```text
cluster
```

分割。

不是 sequence。

---

# 15. Stratified Group Split

需要尽可能维持：

```text
IS family distribution
taxonomy distribution
protein length distribution
```

因此优化目标可以设为：

```text
minimize:

family_distribution_difference
+
taxonomy_distribution_difference
+
length_distribution_difference
```

但第一约束必须始终是：

```text
cluster integrity
```

不能为了平衡 family 拆 cluster。

---

# 16. Family-aware splitting

只有 `<30% identity test` 仍不足够。

建议设计四套独立 benchmark。

---

## Test-A：Cluster-held-out

要求：

```text
test Tpase
≤30% identity
to training Tpases
```

但：

```text
family 可以在 train 出现
```

回答：

> 能否发现同一个 IS family 内的远缘成员？

---

## Test-B：Family-held-out

选择若干完整 IS family：

```text
Family X
Family Y
Family Z
```

完全不进入：

```text
training
validation
```

只用于：

```text
test
```

回答：

> 能否识别训练过程中完全没有见过的 IS family？

这是 novel-family generalization benchmark。

---

# 17. Leave-one-family-out benchmark

如果 family 数量允许：

```text
Train:
all families except IS256

Test:
IS256
```

依次：

```text
IS1
IS3
IS4
IS5
...
```

得到：

```text
family-specific zero-shot recall
```

这是非常有价值的结果。

---

# 18. Taxonomy-held-out benchmark

另建：

```text
Taxon-held-out
```

例如：

```text
Train:
多数 bacterial phyla

Test:
selected genus/species/phylum
```

避免：

```text
same host lineage
+
same IS lineage
```

造成隐性泄漏。

可分别测试：

```text
genus-held-out
species-held-out
phylum-held-out
```

---

# 19. Genome-held-out benchmark

完整 genome evaluation 必须：

> 一个 genome 的所有 IS 都只属于一个 split。

禁止：

```text
same genome
→ some IS train
→ other IS test
```

因为 flanking sequence、copy number 和 genome-specific composition 都可能造成 leakage。

---

# 20. Nucleotide-level leakage

除了 Tpase，还必须检查完整 IS nucleotide similarity。

第二轮聚类：

```text
full IS DNA
```

例如：

```text
80%
90%
95%
```

identity clustering。

对于 strict remote benchmark：

```text
test full IS
不得与 train full IS 高度相似
```

建议最低要求：

```text
<80% nucleotide identity
over ≥80% alignment coverage
```

---

# 21. 推荐的泄漏检查顺序

```text
Raw ISfinder
    ↓
exact duplicate collapse
    ↓
Tpase 30% clustering
    ↓
IS DNA similarity clustering
    ↓
genome grouping
    ↓
taxonomy grouping
    ↓
train / val / test
    ↓
ALL-vs-ALL leakage audit
```

最终必须生成：

```text
leakage_report.tsv
```

---

# 22. Leakage audit

必须检查：

```text
max protein identity:
train ↔ test

max DNA identity:
train ↔ test
```

并输出：

```tsv
test_id  nearest_train_id  protein_identity  coverage  dna_identity
```

如果严格 test 出现：

```text
protein identity >30%
AND
coverage ≥80%
```

则 benchmark 构建失败。

CI 应直接：

```text
exit 1
```

---

# 23. PLM-specific data leakage

PLM 带来一个无法完全消除的问题：

> ESM/ProtT5 等 foundation model 本身可能已经在包含部分 ISfinder homolog 的大型公共蛋白库上预训练。

因此必须区分：

```text
task-specific leakage
```

和：

```text
foundation-model pretraining exposure
```

后者通常无法完全追溯。

论文中必须明确声明。

因此重点保证：

```text
DeepISE supervised training dataset
```

本身严格 cluster-disjoint。

---

# 24. Negative dataset 是整个项目第二重要的问题

随机抽取普通 bacterial protein 作为 negative 会使任务过于简单：

```text
ribosomal protein
vs
Tpase
```

PLM 很容易区分。

实际挑战是：

```text
Tpase
vs
Tpase-like proteins
vs
other mobile-element proteins
```

因此 negative dataset 必须分层构建。

---

# 25. Negative-Level-1：Easy negatives

从高质量 bacterial / archaeal proteome 采样：

```text
ribosomal proteins
metabolic enzymes
transport proteins
housekeeping proteins
```

要求：

```text
无 transposase annotation
无 IS-associated domain
```

比例：

```text
20–30%
```

---

# 26. Negative-Level-2：Length-matched negatives

Tpase 常有特定长度分布。

如果：

```text
positive median length = 320 aa
negative median length = 90 aa
```

模型可能直接学习 length。

因此 negative 必须进行：

```text
protein-length matching
```

例如每一个 positive：

```text
length = L
```

negative：

```text
0.8L ≤ length ≤ 1.2L
```

---

# 27. Negative-Level-3：Taxonomy-matched negatives

尽量从 positive IS 的同一个：

```text
species
genus
genome
```

获取普通 protein。

例如：

```text
E. coli IS Tpase
```

对应 negatives：

```text
E. coli proteins
```

而不是：

```text
eukaryotic proteins
```

---

# 28. Negative-Level-4：Hard negatives

这是模型真正价值所在。

必须大量包含：

```text
integrase
recombinase
resolvase
invertase
DNA nuclease
DNA repair protein
helicase
RNase H-like protein
DDE nuclease
HUH nuclease
replication protein
phage integrase
plasmid mobility proteins
site-specific recombinase
retroelement-associated proteins
```

特别是：

```text
DDE-family proteins
```

必须作为重点 hard negative。

因为很多 Tpase 本身具有：

```text
DDE catalytic fold
```

---

# 29. Mobile-element hard negatives

建议单独建立：

```text
MGE-decoy set
```

包括：

```text
ICE-associated proteins
integron integrases
phage integrases
plasmid recombinases
large transposon proteins
conjugation proteins
```

注意：

> 某些 mobile-element proteins 与 IS Tpase 生物学边界本身就不绝对。

因此不要强行全部作为训练 negative。

建议分成：

```text
verified_negative
ambiguous_mobile
```

后者只用于：

```text
challenge benchmark
```

而不进入 binary training。

---

# 30. Negative-Level-5：Genomic-context decoys

Stage 2 还需要 element-level negatives。

例如找到：

```text
protein
+
flanking inverted repeats
```

但不是真实 IS 的 genomic region。

可从 genome 中随机生成：

```text
length-matched genomic windows
```

并匹配：

```text
GC content
ORF count
repeat density
window length
```

这些是：

```text
context negatives
```

---

# 31. Hard-negative mining

第一版模型训练后：

```text
model v0
   ↓
scan large bacterial proteome
   ↓
high-confidence predictions
   ↓
remove known Tpases
   ↓
manual / domain inspection
   ↓
false positives
```

将这些：

```text
false-positive proteins
```

加入：

```text
hard-negative-v2
```

重新训练。

这是非常重要的一轮。

---

# 32. Negative ratio

训练阶段不要强行模拟真实 prevalence。

推荐实验：

```text
positive : negative

1 : 1
1 : 3
1 : 5
1 : 10
```

训练可以：

```text
1:3
```

但 test 应增加：

```text
realistic proteome benchmark
```

否则 precision 没意义。

---

# 33. Protein-level模型的第一阶段 baseline

不要一开始 fine-tune ESM。

先建立简单 baseline。

### B0

```text
BLASTP
```

---

### B1

```text
MMseqs2
```

---

### B2

```text
HMMER
whole-Tpase pHMM
```

模拟 ISEScan 思路。

---

### B3

```text
catalytic-domain HMM
```

模拟 digIS 思路。

---

### B4

```text
TnpDiscovery-like
classical ML
```

---

### B5

```text
PLM embedding
+
simple classifier
```

例如：

```text
ESM2 embedding
        ↓
mean pooling
        ↓
logistic regression
```

先不要神经网络。

---

# 34. 为什么 Logistic Regression 很重要

如果：

```text
frozen ESM embedding
+
linear classifier
```

已经显著超过：

```text
pHMM
```

则说明：

> PLM representation 本身包含额外 biological information。

这比：

```text
大型复杂 neural network
```

更容易解释、发表和复现。

---

# 35. 推荐 PLM baseline

第一阶段最多测试：

```text
ESM-2
ProtT5
```

可增加一个 structure-aware model。

不要一开始同时测试十几个 PLM。

---

# 36. PLM 模型方案

### Model P1

```text
ESM2
 ↓
mean embedding
 ↓
Logistic Regression
```

### Model P2

```text
ESM2
 ↓
embedding
 ↓
MLP
```

### Model P3

```text
ProtT5
 ↓
embedding
 ↓
MLP
```

### Model P4

```text
ESM2 fine-tuning
```

只有 P1/P2 确认有效后才做 P4。

---

# 37. Structure 模块

不建议所有 ORF 首先跑 AlphaFold。

结构 pipeline：

```text
protein candidates
       ↓
sequence / PLM prescreen
       ↓
top candidates
       ↓
structure-aware search
       ↓
Foldseek
```

---

# 38. Tpase Structure Reference Database

从高质量 ISfinder Tpase 建立：

```text
TpaseStructureDB
```

结构来源优先级：

```text
1. PDB experimental
2. AlphaFoldDB existing prediction
3. local structure prediction
```

保存：

```yaml
tpase_id:
is_family:
structure_source:
structure_file:
plddt:
foldseek_cluster:
catalytic_domain:
```

---

# 39. Structure clustering

Foldseek 聚类：

```text
known Tpase structures
        ↓
structural clusters
```

研究：

```text
sequence cluster
vs
structure cluster
vs
IS family
```

一个很有意义的问题是：

> Tpase 在 `<30% sequence identity` 后是否仍保留 family-specific structural clusters？

---

# 40. Structure features

候选 feature：

```text
best Foldseek E-value
best TM-score
alignment coverage
3Di similarity
structure-cluster ID
catalytic-domain structural score
```

不要只保存：

```text
best hit
```

---

# 41. Evidence Fusion

最终 Tpase-level evidence：

```text
E_seq
E_hmm
E_plm
E_structure
E_motif
```

先使用可解释模型：

```text
Logistic Regression
XGBoost / LightGBM
```

而不是深度 fusion network。

例如：

```text
P(Tpase) =
f(
  HMM_score,
  PLM_similarity,
  PLM_classifier_probability,
  Foldseek_TMscore,
  motif_score
)
```

---

# 42. Stage 2：IS boundary inference

找到 Tpase 后：

```text
candidate Tpase
       ↓
extract ±N kb
```

推荐 initial：

```text
±10 kb
```

但具体 family 可设 family-dependent window。

---

# 43. Boundary evidence

需要寻找：

```text
TIR
TSD
known IS similarity
multiple-copy alignment
element length
ORF architecture
```

---

# 44. TIR detection

Rust 实现。

搜索：

```text
upstream flank
vs
reverse-complement downstream flank
```

支持：

```text
mismatch
small gap
variable TIR length
```

输出：

```yaml
tir:
  left:
  right:
  length:
  identity:
  mismatches:
  gaps:
  score:
```

---

# 45. TSD detection

预测 candidate boundary 后比较：

```text
left outside boundary
right outside boundary
```

寻找：

```text
2–20 bp
```

短 direct repeat。

输出：

```yaml
tsd:
  sequence:
  length:
  confidence:
```

注意：

> 没有 TSD 不代表不是 IS。

不同 IS family 的 transposition mechanism 不同。

因此 TSD 只能作为 positive evidence。

---

# 46. ORF architecture

保存：

```text
ORF count
ORF orientation
protein lengths
inter-ORF distance
overlap
frameshift
```

例如：

```text
IS200/IS605
```

与典型 DDE family 的 architecture 不同。

不能建立“所有 IS 都是一 ORF + 两个 TIR”的统一假设。

---

# 47. IS-level scoring

建议：

```text
IS_score =
f(
  Tpase_score,
  TIR_score,
  TSD_score,
  length_score,
  ORF_architecture,
  DNA_similarity,
  copy_evidence
)
```

第一版继续使用：

```text
interpretable gradient boosting
```

或 logistic model。

---

# 48. IS candidate classification

建议至少六类：

```text
KNOWN_IS
REMOTE_FAMILY_MEMBER
PUTATIVE_NOVEL_IS
PARTIAL_IS
DEGENERATED_IS
TPASE_ONLY
```

增加：

```text
UNCERTAIN
```

避免强制分类。

---

# 49. Known / Remote / Novel 定义

不要根据：

```text
模型置信度
```

定义 novel。

建议操作性定义：

### Known IS

与 ISfinder：

```text
high nucleotide similarity
```

例如：

```text
≥80% identity
≥80% coverage
```

具体阈值需要 benchmark 后确定。

### Remote family member

```text
low DNA similarity
+
detectable Tpase-family evidence
+
consistent structure/context
```

### Putative novel IS

```text
no strong DNA match
+
low/no classical protein homology
+
PLM/structure evidence
+
credible IS genomic architecture
```

正式论文里建议始终写：

```text
putative novel
```

除非实验验证。

---

# 50. Benchmark 总体设计

DeepISE 必须同时做：

```text
Protein-level benchmark
        +
Element-level benchmark
        +
Genome-level benchmark
        +
Remote-homology benchmark
```

只做 protein AUROC 不足以证明 IS discovery。

---

# 51. Benchmark 1：Tpase classification

方法：

```text
BLASTP
MMseqs2
whole-Tpase pHMM
digIS catalytic-domain pHMM
TnpDiscovery
ESM2-linear
ESM2-MLP
ProtT5
Foldseek
DeepISE fusion
```

---

# 52. Protein-level metrics

主要指标：

```text
AUPRC
AUROC
Precision
Recall
F1
MCC
Specificity
```

其中：

> **AUPRC 优先级高于 AUROC。**

因为 realistic genome 中：

```text
Tpase << non-Tpase
```

类别高度不平衡。

---

# 53. Fixed-FDR benchmark

更有实际意义：

```text
Recall @ 1% FDR
Recall @ 5% FDR
Recall @ 10% FDR
```

回答：

> 在只允许很少假阳性的情况下能找回多少远缘 Tpase？

---

# 54. Identity-stratified benchmark

这是 DeepISE 的核心 Figure。

分箱：

```text
>70%
50–70%
30–50%
20–30%
<20%
```

每组计算：

```text
Recall
Precision
F1
```

目标 Figure：

```text
Recall
1.0 ┤
    │                         DeepISE
0.8 ┤                      ───────────
    │
0.6 ┤              PLM / Foldseek
    │
0.4 ┤         HMM
    │
0.2 ┤ BLAST
    │
0.0 └──────────────────────────────
       >70  50–70 30–50 20–30 <20
              sequence identity
```

---

# 55. Macro-family metrics

不能让：

```text
IS3
IS5
```

等大 family 主导结果。

报告：

```text
Micro F1
Macro F1
per-family recall
```

尤其：

```text
Macro Recall
```

非常重要。

---

# 56. Benchmark 2：完整 IS detection

主要 baseline：

```text
ISEScan
digIS
Palidis
DeepISE
```

如果条件允许：

```text
OASIS
ISsaga
```

作为附加历史 baseline。

---

# 57. 为什么 Palidis 必须 benchmark

Palidis 的优势是：

```text
repeat / terminal-structure driven
```

与 DeepISE：

```text
Tpase + multimodal protein evidence
```

路线不同。

因此 Palidis 可以检验：

> DeepISE 是否只是在重复另一种 novel-IS detector？

---

# 58. IS-level True Positive 定义

建议同时报告宽松和严格指标。

## Detection TP

如果 prediction 与 truth：

```text
reciprocal overlap ≥80%
```

记为：

```text
TP-detection
```

---

# 59. Strict Boundary TP

更加严格：

```text
|predicted_start - true_start| ≤ 20 bp

AND

|predicted_end - true_end| ≤ 20 bp
```

称：

```text
TP-boundary20
```

另外报告：

```text
±5 bp
±10 bp
±20 bp
±50 bp
```

---

# 60. Boundary error

连续指标：

```text
start_error_bp
end_error_bp
mean_boundary_error
median_boundary_error
```

这是非常重要的指标。

因为：

```text
找到 Tpase
```

和：

```text
准确找到 IS
```

是两件不同的事。

---

# 61. Element-level metrics

报告：

```text
Sensitivity / Recall
Precision
F1
FDR
MCC
```

以及：

```text
Strict boundary accuracy
```

---

# 62. Family classification benchmark

对 detection 正确的 candidate：

```text
family_accuracy
macro_family_F1
top-3 family accuracy
```

另设：

```text
UNKNOWN
```

允许模型拒绝分类。

对 novel family：

> 输出 UNKNOWN 是正确行为，不应该强制归到最近 family。

---

# 63. Partial IS benchmark

完整和 partial IS 必须分开。

报告：

```text
Complete IS recall
Partial IS recall
Degenerated IS recall
```

不要混成一个 sensitivity。

---

# 64. Benchmark 3：ISbrowser curated genomes

ISEScan 原始论文使用 ISbrowser 手工注释 genome benchmark，因此 DeepISE 应尽可能保留这一历史 benchmark，用于与已有文献横向比较。

但：

> 如果 ISbrowser 中的 IS 被用于训练，则不能再作为 test。

因此必须选择：

```text
TRAINING
```

或：

```text
HISTORICAL BENCHMARK
```

二者只能选一个用途。

推荐：

> ISbrowser benchmark genome 完全从训练数据中剔除。

---

# 65. Benchmark genome contamination prevention

所有 benchmark genome：

```text
Genome A
```

中的 IS：

```text
不得进入 train
```

进一步检查其 Tpase：

```text
不得与 train >30% identity
```

如果做 strict remote benchmark。

---

# 66. Benchmark 4：Remote IS benchmark

这是 DeepISE 最关键的数据集。

定义：

```text
test Tpase
≤30% identity
to every training Tpase
```

进一步划分：

```text
20–30%
<20%
```

比较：

```text
ISEScan
digIS
DeepISE
```

Palidis 可做 element-level comparison。

---

# 67. Benchmark 5：Family-held-out novel simulation

人为模拟：

```text
novel family
```

例如：

```text
remove Family X completely
```

训练 DeepISE。

然后：

```text
scan Family X genomes
```

评价：

```text
Tpase recall
IS recall
UNKNOWN classification rate
false family assignment rate
```

这是最接近“发现未知 IS family”的可控实验。

---

# 68. Open-set evaluation

Novel IS discovery 本质是：

```text
open-set recognition
```

因此必须报告：

```text
known-family accuracy
unknown-family detection
false known-family assignment
```

推荐指标：

```text
AUROC_unknown
AUPRC_unknown
Open-set F1
```

---

# 69. Benchmark 6：Realistic proteome scan

选择完全没有用于训练的：

```text
100–1000 bacterial genomes
```

运行：

```text
DeepISE
ISEScan
digIS
Palidis
```

统计：

```text
predictions / Mb
runtime
memory
known IS recovery
novel candidate count
```

---

# 70. Novel candidate quality

不能说：

```text
DeepISE 找到更多 novel IS
```

就代表更好。

对 DeepISE-only candidate 建立 evidence score。

例如：

```text
Tpase PLM
Tpase structure
catalytic motif
TIR
TSD
copy evidence
ORF architecture
```

统计：

```text
≥3 independent evidence channels
≥4
≥5
```

---

# 71. Consensus analysis

生成：

```text
ISEScan ∩ digIS ∩ Palidis ∩ DeepISE
```

以及：

```text
DeepISE only
```

UpSet plot 很适合。

重点人工分析：

```text
DeepISE-only
+
high-confidence
```

的候选。

---

# 72. Ablation study

DeepISE 如果是 multimodal framework，必须做消融。

模型：

```text
M0 HMM
M1 PLM
M2 Structure
M3 HMM + PLM
M4 HMM + Structure
M5 PLM + Structure
M6 HMM + PLM + Structure
M7 + genomic context
M8 + TIR/TSD
```

比较：

```text
AUPRC
remote recall
element precision
```

这样才能回答：

> 每一种 evidence 到底贡献了什么？

---

# 73. 最重要的消融结果

理想结果应该是：

```text
Protein-level:

PLM / Structure
    ↑
remote recall


Element-level:

TIR / TSD / context
    ↑
precision
```

即：

```text
AI解决 sensitivity
genomic biology解决 specificity
```

这是非常漂亮的方法学故事。

---

# 74. Calibration

输出 probability 必须 calibration。

使用：

```text
Platt scaling
isotonic regression
```

评价：

```text
Brier score
ECE
reliability diagram
```

最终 DeepISE 才能合理输出：

```text
High confidence
Medium confidence
Low confidence
```

---

# 75. 推荐 confidence scheme

例如：

```text
HIGH
P(IS) ≥0.95

MEDIUM
0.80–0.95

LOW
0.50–0.80
```

阈值不能拍脑袋。

必须通过 validation：

```text
FDR-controlled
```

确定。

---

# 76. Statistical comparison

工具之间不能只比较一个数字。

对 genome-level metric 使用：

```text
bootstrap confidence interval
```

例如：

```text
10,000 bootstrap replicates
```

报告：

```text
ΔRecall
95% CI
```

对于 paired genome benchmark，可使用：

```text
paired bootstrap
```

---

# 77. Runtime benchmark

因为 DeepISE 最终要做 Rust 工具，性能也是结果之一。

记录：

```text
wall time
CPU time
peak RSS
threads
GPU memory
```

标准化：

```text
seconds / Mb genome
```

或：

```text
genomes / CPU-hour
```

---

# 78. 两套 DeepISE 模式

建议最终产品设计为：

## DeepISE-fast

```text
MMseqs/HMM
+
PLM prescreen
+
Rust context
```

无需结构预测。

适合：

```text
10,000+ genomes
```

---

## DeepISE-deep

```text
HMM
+
PLM
+
Foldseek
+
optional structure prediction
+
TIR/TSD
```

适合：

```text
novel IS discovery
```

---

# 79. 推荐最终 CLI

```bash
deepise scan genome.fna
```

默认：

```text
fast mode
```

---

```bash
deepise scan genome.fna --mode deep
```

深度模式。

---

```bash
deepise protein proteins.faa
```

只做 Tpase prediction。

---

```bash
deepise benchmark benchmark.yaml
```

运行 benchmark。

---

# 80. Rust / Python 通讯方式

第一阶段不要 PyO3 深度耦合。

建议：

```text
Rust
 ↓
Parquet / Arrow / JSONL
 ↓
Python
 ↓
prediction.parquet
 ↓
Rust
```

研究阶段这样：

- 调试容易；
- 可替换模型；
- Python/Rust 可以单独运行；
- 数据容易复现。

模型稳定后再考虑：

```text
PyO3
```

或：

```text
ONNX
```

---

# 81. 推荐中间格式

大数据：

```text
Parquet
```

小型结果：

```text
JSONL
```

最终：

```text
GFF3
TSV
FASTA
JSON
```

---

# 82. Repository 结构

```text
DeepISE/
│
├── Cargo.toml
├── README.md
├── LICENSE
│
├── crates/
│   ├── deepise-core/
│   ├── deepise-genome/
│   ├── deepise-tir/
│   ├── deepise-tsd/
│   └── deepise-cli/
│
├── python/
│   └── deepise_ml/
│       ├── dataset/
│       ├── embeddings/
│       ├── models/
│       ├── structure/
│       ├── training/
│       └── evaluation/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── benchmark/
│
├── models/
│
├── workflow/
│   ├── Snakefile
│   └── config.yaml
│
├── benchmark/
│
├── scripts/
│
├── tests/
│
└── docs/
    ├── DATASET.md
    ├── BENCHMARK.md
    ├── MODEL.md
    └── METHODS.md
```

---

# 83. 数据集 manifest

每个 dataset 都必须：

```yaml
name: deepise-v1
version: 1
created_at:
source_snapshot:

positive:
  count:
  families:

negative:
  easy:
  matched:
  hard:
  ambiguous:

split:
  strategy: cluster30
  train:
  validation:
  test:

leakage:
  max_train_test_identity:
  coverage_threshold:
```

---

# 84. 可重复性

记录：

```text
ISfinder snapshot
NCBI accessions
software versions
model checkpoints
random seed
MMseqs parameters
HMMER parameters
Foldseek parameters
training hyperparameters
GPU
```

所有 experiment：

```text
config.yaml
```

驱动。

---

# 85. 推荐 Phase 0：Data feasibility

目标：

> 确认数据够不够训练。

工作：

```text
ISfinder extraction
Tpase extraction
family statistics
sequence length
family imbalance
30% clustering
```

最关键输出：

```text
多少条 IS？
多少 Tpase？
30% clustering 后还剩多少独立 cluster？
每个 family 有多少 cluster？
```

如果某 family：

```text
500 proteins
```

但是：

```text
30% cluster = 4
```

说明表面数据量虚高。

---

# 86. Phase 1：Go / No-Go Experiment

这是整个项目最重要阶段。

只做：

```text
Tpase remote-homology detection
```

不做完整 DeepISE。

比较：

```text
BLAST
MMseqs2
pHMM
ESM2
ProtT5
Foldseek
```

数据：

```text
strict cluster30 test
```

---

# 87. Go 标准

建议提前制定。

例如 `<30% identity` test：

PLM / Structure / Fusion 相比最佳 HMM：

```text
Recall@5%FDR
absolute improvement ≥10%
```

或：

```text
AUPRC improvement ≥10%
```

且至少：

```text
多个 IS families
```

均存在提升。

则：

```text
GO
```

进入完整工具开发。

---

# 88. No-Go 条件

如果：

```text
PLM ≈ HMM
Foldseek ≈ HMM
Fusion ≈ HMM
```

尤其：

```text
<30% identity
```

没有明显提升，

则不应该为了“AI”继续开发复杂系统。

可以转而研究：

```text
TIR/TSD
WGS insertion calling
```

等更有价值方向。

---

# 89. Phase 2：Protein model

建立：

```text
DeepISE-Tpase
```

目标：

```text
high-sensitivity remote Tpase detector
```

完成：

```text
embedding
classifier
calibration
family/open-set output
```

---

# 90. Phase 3：Genome-context engine

Rust 开发：

```text
candidate mapping
TIR
TSD
boundary
copy search
context scoring
```

目标：

```text
Tpase
→
complete IS
```

---

# 91. Phase 4：Complete benchmark

比较：

```text
ISEScan
digIS
Palidis
DeepISE-fast
DeepISE-deep
```

主要 endpoint：

```text
IS Recall
IS Precision
Boundary accuracy
Remote IS Recall
Novel simulation Recall
```

---

# 92. Phase 5：Real biological validation

强烈建议加入真实 WGS insertion event。

例如：

```text
BL21 parent
vs
Cas9 edited clone
```

寻找：

```text
DeepISE candidate IS
       ↓
new insertion site
       ↓
split reads
discordant reads
       ↓
TSD
       ↓
experimental / WGS-supported event
```

这能够把：

```text
computational prediction
```

升级为：

```text
biologically validated transposition
```

---

# 93. DeepISE 与用户现有研究的结合

未来可以增加独立模块：

```bash
deepise call \
    --reference BL21.fasta \
    --control WT.bam \
    --sample edited.bam
```

输出：

```text
new IS insertion
IS identity
breakpoint
TSD
junction support
read depth
confidence
```

但这是：

```text
DeepISE v2
```

不要进入 MVP。

---

# 94. 推荐论文 Figure

## Figure 1

DeepISE architecture：

```text
Genome
 ↓
Tpase discovery
 ↓
HMM + PLM + Structure
 ↓
TIR/TSD/context
 ↓
IS
```

---

## Figure 2

Dataset construction：

```text
ISfinder
 ↓
deduplicate
 ↓
30% cluster
 ↓
cluster-aware split
```

---

## Figure 3

Remote homolog benchmark：

```text
Recall vs sequence identity
```

这是最重要 Figure。

---

## Figure 4

Ablation：

```text
HMM
PLM
Structure
Fusion
Fusion + context
```

---

## Figure 5

Genome-level comparison：

```text
ISEScan
digIS
Palidis
DeepISE
```

---

## Figure 6

Novel IS examples：

```text
Tpase structure
+
PLM similarity
+
TIR
+
TSD
+
genomic architecture
```

---

## Figure 7

真实 biological case：

```text
new insertion
+
WGS junction
+
TSD
```

---

# 95. 论文主要 benchmark 表

建议最终形成：

| Method | Known Recall | Remote Recall | Precision | Boundary ±20 bp | Family Macro-F1 | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| ISEScan | | | | | | |
| digIS | | | | | | |
| Palidis | | | | | | |
| DeepISE-fast | | | | | | |
| DeepISE-deep | | | | | | |

---

# 96. 远缘 benchmark 表

| Method | >70% | 50–70% | 30–50% | 20–30% | <20% |
|---|---:|---:|---:|---:|---:|
| BLASTP | | | | | |
| MMseqs2 | | | | | |
| pHMM | | | | | |
| digIS-like HMM | | | | | |
| ESM2 | | | | | |
| ProtT5 | | | | | |
| Foldseek | | | | | |
| DeepISE | | | | | |

指标：

```text
Recall @ fixed FDR
```

---

# 97. Benchmark 公平性

所有工具：

```text
same genome
same FASTA
same CPU allocation
recommended/default parameters
fixed software version
```

记录：

```text
version
command
container
runtime
```

使用：

```text
Docker / Apptainer
```

固定环境。

---

# 98. 不允许针对 benchmark 调参数

例如：

```text
ISEScan default
digIS default/recommended
Palidis recommended
DeepISE threshold
```

DeepISE threshold 只能：

```text
validation set
```

决定。

绝不能看 test 后调整。

---

# 99. 最重要的 methodological safeguards

DeepISE 必须明确执行以下规则：

1. **禁止 random sequence split。**
2. **Tpase train/test ≤30% identity。**
3. **完整 genome 不跨 split。**
4. **test genome IS 不进入 training。**
5. **benchmark threshold 只用 validation 调整。**
6. **negative 必须包含 hard negatives。**
7. **partial IS 与 complete IS 分开评价。**
8. **known 与 novel benchmark 分开。**
9. **必须报告 precision，不只 recall。**
10. **必须做 identity-stratified evaluation。**
11. **必须做 family macro metrics。**
12. **必须做 multimodal ablation。**
13. **PLM pretraining exposure 必须声明。**
14. **DeepISE-only candidate 不能自动宣称 novel IS。**

---

# 100. MVP 不应该包含什么

第一版不要：

```text
自研 foundation model
自研 AlphaFold
自研 Foldseek
GPU Rust inference
Web frontend
database server
LLM agent
distributed training
```

这些都不是论文的核心问题。

---

# 101. 推荐 MVP

```text
DeepISE v0.1

ISfinder dataset builder
        +
30% cluster split
        +
negative dataset
        +
BLAST/MMseq/HMM baseline
        +
ESM2 embedding classifier
        +
Foldseek benchmark
```

先回答：

> AI/structure 到底有没有增益？

---

# 102. DeepISE v0.2

如果 Go：

```text
DeepISE-Tpase
        +
Rust genomic context
        +
TIR detector
        +
TSD detector
        +
boundary inference
```

---

# 103. DeepISE v1.0

```text
DeepISE-fast
DeepISE-deep
GFF3/TSV/JSON
complete benchmark
container
Bioconda
paper dataset
```

---

# 104. 推荐开发顺序

```text
01 ISfinder snapshot
02 parser
03 dataset QC
04 exact dedup
05 30% clustering
06 leakage audit
07 negative-set construction
08 BLAST baseline
09 MMseqs baseline
10 HMM baseline
11 ESM2 embedding
12 linear classifier
13 Foldseek benchmark
14 Go / No-Go decision
15 multimodal fusion
16 Rust TIR
17 Rust TSD
18 boundary inference
19 complete IS benchmark
20 novel-family benchmark
21 real genome validation
```

---

# 105. 项目的真正创新点

DeepISE 不应定位成：

> “使用深度学习识别 IS。”

而应定位：

> **DeepISE is a multimodal framework for remote insertion-sequence discovery that combines transposase sequence profiles, protein-language-model representations, structural homology, and genomic-context evidence.**

更具体地：

```text
sequence
   │
PLM
   │
structure
   │
catalytic motif
   │
   ▼
remote Tpase discovery
   │
   ▼
TIR + TSD + context
   │
   ▼
complete IS discovery
```

---

# 106. 最重要的科学假设

整个项目实际上可以浓缩成：

> **Transposase sequence diverges faster than its functional representation and catalytic structure.**

因此：

```text
very low sequence identity
        │
        ▼
classical homology fails
        │
        ▼
PLM / structure retains signal
        │
        ▼
remote Tpase recovered
        │
        ▼
genomic-context evidence
        │
        ▼
remote IS recovered
```

如果这个假设能够通过严格 `<30% identity` benchmark 被证明，DeepISE 就具备明显的方法学价值。

---

# 107. 最关键的 Go / No-Go 实验

项目启动后第一篇内部报告只需要回答：

```text
Question:

On cluster-disjoint Tpases
with <30% sequence identity
to the training/reference set,

does:

PLM
or
Foldseek
or
PLM + Foldseek

recover significantly more true Tpases
than:

HMMER / digIS-style pHMM

at the same FDR?
```

如果答案：

```text
YES
```

DeepISE 值得继续。

如果答案：

```text
NO
```

停止复杂 PLM 开发。

---

# 108. 最终成功标准

DeepISE v1 可以认为成功，需要同时满足：

```text
1. <30% identity Tpase recall
   显著优于传统 HMM baseline

2. genome-level IS precision
   不因增加 remote candidates 明显下降

3. complete IS boundary accuracy
   达到可实际注释水平

4. family-held-out benchmark
   能识别部分 unseen-family elements

5. DeepISE-only candidates
   有独立 TIR/TSD/structure/context evidence

6. Runtime
   可以用于批量 microbial genome analysis
```

---

# 109. 最推荐的核心技术组合

第一阶段：

```text
MMseqs2
+
HMMER
+
ESM2
+
Foldseek
+
scikit-learn
```

第二阶段：

```text
Rust
+
needletail
+
rust-bio
+
rayon
```

模型成熟后：

```text
PyTorch → ONNX
```

可以研究是否把轻量 classifier 直接嵌入 Rust。

---

# 110. 一句话项目路线

```text
ISfinder curated data
        ↓
30%-identity leakage-free benchmark
        ↓
HMM vs PLM vs Foldseek
        ↓
prove remote-homology gain
        ↓
Rust TIR/TSD/context engine
        ↓
complete IS discovery
        ↓
ISEScan / digIS / Palidis benchmark
        ↓
novel-family discovery
        ↓
WGS biological validation
```

---

# 111. 项目优先级

## P0 — 必须

- ISfinder dataset
- exact deduplication
- 30% cluster split
- leakage audit
- hard-negative dataset
- HMM baseline
- PLM baseline
- identity-stratified benchmark

## P1 — 核心创新

- Foldseek / structure evidence
- multimodal fusion
- family-held-out benchmark
- open-set classification

## P2 — 完整工具

- Rust genomic engine
- TIR
- TSD
- boundary inference
- GFF3
- ISEScan/digIS/Palidis comparison

## P3 — 论文增强

- large genome survey
- putative novel IS
- structure examples
- WGS transposition validation

---

# 112. 最终原则

DeepISE 的开发始终遵循三个原则：

### Principle 1 — Benchmark before engineering

先证明：

```text
PLM / structure > HMM
```

再开发大型 Rust pipeline。

### Principle 2 — Split by homology, not randomly

```text
≤30% identity
```

必须成为项目最核心的数据学约束。

### Principle 3 — Tpase is evidence, not an IS

只有：

```text
protein evidence
+
DNA boundary
+
genomic context
```

整合之后，才能称作：

```text
Insertion Sequence prediction
```

而不是：

```text
transposase prediction
```

---

# 113. 建议论文题目

第一选择：

**DeepISE: multimodal discovery of remote insertion sequences using protein language models, structural homology and genomic context**

如果突出原核生物：

**DeepISE: protein language model- and structure-assisted discovery of insertion sequences in prokaryotic genomes**

如果最终 novel discovery 很强：

**Multimodal protein representations enable remote insertion-sequence discovery in prokaryotic genomes**

---

# 114. 最终研究主线

DeepISE 最终应该形成一个非常清晰的科学故事：

```text
Known IS
   ↓
sequence similarity works

Remote IS
   ↓
sequence similarity decreases

Tpase catalytic function
   ↓
PLM representation retained

Protein fold
   ↓
structural similarity retained

Genomic context
   ↓
TIR / TSD / architecture retained

             ↓

DeepISE integrates all evidence

             ↓

high-sensitivity
remote IS discovery

without unacceptable
false-positive inflation
```

这才是 DeepISE 相比 ISEScan、digIS、Palidis 和单纯 ML transposase classifier 最核心的区别。
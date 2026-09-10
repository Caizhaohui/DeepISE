# DeepISE Phase-0 / Phase-1 Implementation Spec

> Implementation specification for a leakage-controlled benchmark of remote transposase discovery using sequence homology, profile HMMs, protein language models and structural similarity.

**Project:** DeepISE  
**Phase:** 0–1  
**Primary stack:** Rust + Python  
**Document type:** Engineering / Research Implementation Specification  
**Primary objective:** Determine whether PLM and structure-aware representations improve detection of remote transposases under strict homology-controlled evaluation.

---

# 1. Scope

本阶段只解决一个问题：

> 在训练/参考数据与测试 Tpase 之间严格控制 sequence identity 后，ESM2 / ProtT5 / Foldseek 是否能在相同 FDR 下，比 BLASTP、MMseqs2 和 profile HMM 找回更多远缘 transposase？

Phase-0 / Phase-1 **不开发完整 IS boundary caller**。

本阶段不实现：

```text
TIR inference
TSD inference
IS boundary prediction
complete/partial IS classification
WGS insertion calling
novel IS publication claims
web frontend
database server
custom foundation model
```

本阶段最终的 Go / No-Go 决策决定是否进入 Phase-2。

---

# 2. Scientific Endpoint

最重要 endpoint：

```text
Recall@5%FDR
```

在：

```text
test Tpase max identity to reference/training set <30%
```

条件下比较：

```text
BLASTP
MMseqs2
whole-Tpase pHMM
domain-pHMM
ESM2
ProtT5
Foldseek
multimodal fusion
```

核心 Go 条件建议：

```text
PLM / Structure / Fusion

vs

best HMM baseline

absolute Recall@5%FDR improvement >= 0.10
```

并且增益不能仅由单一大 IS family 驱动。

同时要求：

```text
Macro-family Recall
```

存在一致改善。

---

# 3. Repository Layout

Gemini 必须首先建立如下目录。

```text
DeepISE/
├── README.md
├── LICENSE
├── Cargo.toml
├── pyproject.toml
├── uv.lock
├── Makefile
├── .gitignore
├── .github/
│   └── workflows/
│       ├── rust-ci.yml
│       └── python-ci.yml
│
├── config/
│   ├── dataset.yaml
│   ├── clustering.yaml
│   ├── split.yaml
│   ├── negatives.yaml
│   ├── embeddings.yaml
│   └── benchmark.yaml
│
├── crates/
│   ├── deepise-core/
│   │   └── src/
│   └── deepise-cli/
│       └── src/
│
├── python/
│   └── deepise_ml/
│       ├── __init__.py
│       ├── cli.py
│       ├── dataset/
│       │   ├── isfinder.py
│       │   ├── normalize.py
│       │   ├── deduplicate.py
│       │   ├── cluster.py
│       │   ├── split.py
│       │   ├── leakage.py
│       │   └── negatives.py
│       ├── embeddings/
│       │   ├── esm2.py
│       │   ├── prott5.py
│       │   └── common.py
│       ├── baselines/
│       │   ├── blast.py
│       │   ├── mmseqs.py
│       │   ├── hmmer.py
│       │   └── foldseek.py
│       ├── models/
│       │   ├── linear.py
│       │   ├── mlp.py
│       │   └── fusion.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── identity_bins.py
│       │   ├── calibration.py
│       │   ├── bootstrap.py
│       │   └── report.py
│       └── schemas/
│           ├── records.py
│           └── validation.py
│
├── scripts/
│   ├── fetch_isfinder.py
│   ├── build_positive_set.py
│   ├── build_negative_set.py
│   ├── cluster_tpases.sh
│   ├── audit_leakage.py
│   ├── embed_esm2.py
│   ├── embed_prott5.py
│   ├── run_baselines.py
│   └── run_benchmark.py
│
├── workflow/
│   └── Snakefile
│
├── data/
│   ├── raw/
│   │   ├── isfinder/
│   │   └── negatives/
│   ├── interim/
│   ├── processed/
│   ├── splits/
│   └── manifests/
│
├── benchmark/
│   ├── results/
│   ├── figures/
│   ├── tables/
│   └── reports/
│
├── tests/
│   ├── fixtures/
│   ├── test_split.py
│   ├── test_leakage.py
│   ├── test_negatives.py
│   └── test_metrics.py
│
└── docs/
    ├── DATASET.md
    ├── SPLITTING.md
    ├── NEGATIVES.md
    ├── BENCHMARK.md
    └── METHODS.md
```

---

# 4. Responsibility Split

## Python

Phase-0 / Phase-1 中 Python 是主开发语言。

负责：

```text
dataset construction
MMseqs orchestration
negative sampling
PLM embeddings
classical ML
metrics
plots
benchmark reports
```

## Rust

Phase-0 / Phase-1 只建立基础骨架。

Rust 负责：

```text
FASTX validation
sequence ID normalization
high-performance future utilities
CLI shell
```

暂时不要把 ML inference 搬进 Rust。

---

# 5. Python Environment

推荐：

```text
Python >=3.11
```

依赖：

```toml
torch
transformers
sentencepiece
scikit-learn
pandas
polars
pyarrow
numpy
scipy
biopython
pydantic
pyyaml
typer
rich
matplotlib
joblib
tqdm
```

外部程序：

```text
MMseqs2
HMMER
BLAST+
Foldseek
```

可选：

```text
Snakemake
```

---

# 6. Rust Dependencies

根 workspace：

```toml
[workspace]
members = [
    "crates/deepise-core",
    "crates/deepise-cli"
]
resolver = "2"
```

推荐 dependencies：

```toml
clap
serde
serde_json
anyhow
thiserror
needletail
rayon
csv
```

Phase-1 不添加：

```text
tch
candle
onnxruntime
```

除非之后确认 Rust-native inference 有必要。

---

# 7. Data Philosophy

必须建立三个概念：

```text
Raw Data
↓
Normalized Records
↓
Frozen Benchmark Dataset
```

Raw 数据不可修改。

所有转换都写入：

```text
data/interim/
```

最终数据：

```text
data/processed/
```

训练/验证/测试：

```text
data/splits/
```

---

# 8. ISfinder Snapshot

禁止直接在训练流程中实时请求不断变化的数据库。

建立：

```text
data/raw/isfinder/<YYYY-MM-DD>/
```

例如：

```text
data/raw/isfinder/2026-09-10/
```

其中保存：

```text
raw_html/
raw_sequences/
raw_metadata/
SOURCE.json
```

`SOURCE.json`：

```json
{
  "source": "ISfinder",
  "retrieved_at": "2026-09-10",
  "dataset_role": "positive_reference",
  "notes": ""
}
```

---

# 9. ISfinder Acquisition Layer

文件：

```text
python/deepise_ml/dataset/isfinder.py
```

职责：

```text
fetch / parse source records
normalize identifiers
extract IS family
extract nucleotide sequence
extract Tpase protein
extract element metadata
```

不要把网页解析逻辑和后续 dataset transformation 混在一个模块。

如果 ISfinder 当前无法可靠自动抓取完整 metadata：

允许：

```text
manual snapshot
+
parser
```

但必须记录来源和版本。

---

# 10. Canonical Positive Record

统一 schema：

```yaml
record_id: ISFINDER_IS1A_001
source: ISfinder
source_id: IS1A

family: IS1
group: null
subgroup: null

is_sequence:
  sequence: ATGC...
  length: 768

tpase:
  protein_id: IS1A_TPASE
  sequence: MKK...
  length: 248
  start: 100
  end: 843
  strand: "+"

terminal_repeat:
  left: null
  right: null

host:
  organism: Escherichia coli
  taxonomy_id: null

quality:
  complete_element: true
  tpase_available: true
  usable_for_protein_benchmark: true
  usable_for_boundary_benchmark: false

references: []
```

---

# 11. Canonical Storage Format

核心 tabular dataset 使用：

```text
Parquet
```

主要文件：

```text
data/processed/positives.parquet
data/processed/tpases.parquet
data/processed/is_elements.parquet
```

序列同时输出 FASTA：

```text
data/processed/tpases.faa
data/processed/is_elements.fna
```

---

# 12. tpases.parquet Schema

至少包含：

```text
tpase_id
is_id
family
group
subgroup
protein_sequence
protein_length
host
source
quality_level
exact_cluster_id
cluster30_id
split
```

后续增加：

```text
max_identity_to_train
nearest_train_id
identity_bin
```

---

# 13. FASTA ID Rules

禁止使用：

```text
spaces
pipes
complex descriptions
```

FASTA header 只允许：

```text
>tpase_000001
```

metadata 不放 FASTA header。

metadata 通过 Parquet 关联。

---

# 14. Stable ID Generation

ID 必须 deterministic。

例如：

```text
tpase_000001
is_000001
```

排序依据：

```text
source
source_id
family
sequence_hash
```

所有序列计算：

```text
SHA256
```

字段：

```text
protein_sha256
dna_sha256
```

---

# 15. Exact Deduplication

文件：

```text
dataset/deduplicate.py
```

对 Tpase：

```text
100% identical amino-acid sequence
```

collapse 成一个 representative。

生成：

```text
data/interim/tpase_exact_clusters.parquet
```

字段：

```text
representative_id
member_id
cluster_size
sequence_sha256
```

完整 IS DNA 同样做一次。

---

# 16. Exact Duplicate Rules

如果：

```text
same protein sequence
different family annotations
```

不能简单保留其中一个 family。

标记：

```text
annotation_conflict = true
```

这类 record：

```text
不进入 family-classification training
```

但可以进入：

```text
binary Tpase classification
```

---

# 17. MMseqs2 30% Clustering

配置：

```yaml
# config/clustering.yaml

tpase:
  min_seq_id: 0.30
  coverage: 0.80
  cov_mode: 0
  cluster_mode: 2
  sensitivity: 7.5

full_is:
  min_seq_id: 0.80
  coverage: 0.80
  cov_mode: 0
```

---

# 18. Tpase Clustering Command

推荐先建立 sensitivity benchmark，再冻结参数。

默认：

```bash
mkdir -p data/interim/mmseqs/tmp

mmseqs createdb \
  data/processed/tpases.faa \
  data/interim/mmseqs/tpase_db

mmseqs cluster \
  data/interim/mmseqs/tpase_db \
  data/interim/mmseqs/tpase_cluster30 \
  data/interim/mmseqs/tmp \
  --min-seq-id 0.30 \
  -c 0.80 \
  --cov-mode 0 \
  --cluster-mode 2 \
  -s 7.5

mmseqs createtsv \
  data/interim/mmseqs/tpase_db \
  data/interim/mmseqs/tpase_db \
  data/interim/mmseqs/tpase_cluster30 \
  data/interim/tpase_cluster30.tsv
```

---

# 19. Cluster Sanity Check

生成：

```text
benchmark/reports/cluster_statistics.md
```

必须包含：

```text
raw Tpase count
exact-deduplicated count
cluster30 count
median cluster size
max cluster size
singleton fraction
families per cluster
clusters per family
```

最重要检查：

```text
一个 cluster 是否包含多个 IS families？
```

如果很多 cluster 跨 family：

需要记录，这是 biological signal，不允许人为拆 cluster。

---

# 20. Coverage Sensitivity Analysis

必须比较：

```text
-c 0.5
-c 0.7
-c 0.8
```

输出：

```text
benchmark/tables/clustering_sensitivity.tsv
```

不要直接假设 0.8 永远最好。

最终论文参数在 Phase-0 后冻结。

---

# 21. Split Algorithm

文件：

```text
dataset/split.py
```

禁止：

```text
random sequence split
```

单位必须是：

```text
cluster30_id
```

---

# 22. Primary Split

目标：

```text
Train      70%
Validation 15%
Test       15%
```

但不是强制 sequence count 精确 70/15/15。

优先级：

```text
1. cluster integrity
2. family balance
3. protein count balance
4. length balance
5. taxonomy balance
```

---

# 23. Split Optimization

推荐实现：

```text
StratifiedGroupShuffleSplit-like
```

但因为 sklearn 对复杂多标签 group balancing 支持有限，可以自行实现 greedy allocator。

每个 cluster 计算：

```text
cluster_size
family composition
median protein length
taxonomy composition
```

然后 greedy assignment 最小化：

```text
split size deviation
+
family distribution divergence
+
length distribution divergence
```

固定：

```text
random_seed = 42
```

---

# 24. Required Splits

至少产生四套 split。

```text
split_cluster30
split_family_holdout
split_taxon_holdout
split_random_control
```

其中：

```text
split_random_control
```

只作为说明 random split 虚高性能的对照实验。

禁止把它当主结果。

---

# 25. Cluster30 Split

输出：

```text
data/splits/cluster30/
├── train.parquet
├── validation.parquet
├── test.parquet
└── manifest.yaml
```

manifest：

```yaml
strategy: cluster30
seed: 42

protein_identity_constraint:
  threshold: 0.30
  coverage: 0.80

counts:
  train: 0
  validation: 0
  test: 0

clusters:
  train: 0
  validation: 0
  test: 0
```

---

# 26. Family-Held-Out Split

配置：

```yaml
# config/split.yaml

family_holdout:
  minimum_clusters_per_family: 5
  minimum_sequences_per_family: 20
```

只选择：

```text
具有足够独立 cluster 的 family
```

做 family-held-out。

输出：

```text
data/splits/family_holdout/
```

---

# 27. Leave-One-Family-Out

额外实现：

```bash
deepise-data split leave-family-out --family IS256
```

输出：

```text
data/splits/lofo/IS256/
```

后续可批量执行。

---

# 28. Taxonomy Holdout

如果 host taxonomy 信息完整：

支持：

```text
species
genus
family
phylum
```

至少先实现：

```text
genus-level holdout
```

如果 ISfinder taxonomy 不足：

Phase-1 可以延期，但代码接口必须预留。

---

# 29. Leakage Audit

文件：

```text
dataset/leakage.py
```

这是强制模块。

必须做到：

```text
test vs train
validation vs train
```

all-vs-all sequence audit。

---

# 30. Protein Leakage Audit

建议使用：

```text
MMseqs2 search
```

而不是 Python pairwise alignment。

命令：

```bash
mmseqs createdb train.faa train_db
mmseqs createdb test.faa test_db

mmseqs search \
  test_db \
  train_db \
  test_vs_train \
  tmp \
  -s 7.5

mmseqs convertalis \
  test_db \
  train_db \
  test_vs_train \
  test_vs_train.tsv \
  --format-output \
  "query,target,fident,alnlen,qcov,tcov,evalue,bits"
```

---

# 31. Leakage Failure Rule

严格 cluster30 benchmark：

如果任意 test protein 满足：

```text
identity > 0.30
AND
max(query_coverage, target_coverage) >= 0.80
```

则：

```text
FAIL
```

CI 退出：

```text
exit code 1
```

---

# 32. leakage_report.tsv

输出字段：

```text
query_id
nearest_train_id
identity
alignment_length
query_coverage
target_coverage
evalue
bitscore
pass
```

同时把：

```text
max_identity_to_train
```

写回 test metadata。

---

# 33. Identity Binning

基于：

```text
nearest reference/train homolog
```

而不是 family label。

固定 bins：

```text
>=70%
50–70%
30–50%
20–30%
<20%
```

但 strict cluster30 test 主要应该落在：

```text
<30%
```

其他 bin 用于 broader/reference-search benchmark。

---

# 34. Negative Dataset Strategy

配置：

```yaml
# config/negatives.yaml

ratio:
  easy: 0.20
  matched: 0.30
  hard: 0.50

length_tolerance: 0.20

exclude_keywords:
  - transposase
  - insertion sequence
  - mobile element protein

hard_negative_keywords:
  - integrase
  - recombinase
  - resolvase
  - invertase
  - nuclease
  - helicase
  - RNase H
  - DNA repair
  - plasmid
  - phage
```

---

# 35. Negative Sources

优先使用：

```text
Swiss-Prot curated proteins
RefSeq representative prokaryotic proteomes
reviewed UniProt proteins
```

如果使用 NCBI/UniProt：

必须保存：

```text
accession
release/version
download date
taxonomy
annotation
```

---

# 36. Negative Classes

negative_type 字段固定：

```text
easy
length_matched
taxonomy_matched
hard
mge_hard
ambiguous
```

---

# 37. Easy Negatives

从明确 housekeeping/metabolic proteins 中抽样。

目的：

```text
sanity check
```

不能作为主要 benchmark negative。

---

# 38. Length-Matched Negative Sampling

对每条 positive：

```text
L = positive protein length
```

优先采样：

```text
0.8L <= negative length <= 1.2L
```

如果候选不足：

逐步扩展：

```text
±30%
```

但必须记录实际：

```text
length_ratio
```

---

# 39. Taxonomy-Matched Negatives

优先级：

```text
same genome
same species
same genus
same family
```

逐级 fallback。

记录：

```text
taxonomy_match_level
```

---

# 40. Hard Negatives

hard negative pool 至少应包括：

```text
integrase
site-specific recombinase
resolvase
invertase
RNase H-like proteins
DDE nucleases
HUH nucleases
DNA repair enzymes
helicases
phage integration proteins
plasmid mobility proteins
ICE-associated proteins
```

这些比普通 proteins 更重要。

---

# 41. Ambiguous Proteins

任何：

```text
possible transposase
uncharacterized mobile protein
putative mobile-element protein
```

不能直接作为 negative。

放入：

```text
ambiguous
```

只用于 challenge set。

---

# 42. Negative Homology Filter

negative 与 positive Tpase 做：

```text
MMseqs2 search
```

如果：

```text
strong homology to known Tpase
```

则移出 verified negative。

建议初始规则：

```text
E-value <= 1e-5
AND
coverage >= 0.5
```

标记：

```text
ambiguous_homology = true
```

---

# 43. Prevent Negative Leakage

negative proteins 同样做 30% clustering。

同一个 negative homology cluster：

```text
不得同时进入 train/test
```

否则 binary classifier 也会泄漏。

---

# 44. Dataset Balance

训练集第一版：

```text
positive : negative = 1 : 3
```

validation：

```text
1 : 3
```

balanced benchmark：

```text
1 : 3
```

另外建立 realistic benchmark：

```text
真实 proteome prevalence
```

两者都必须报告。

---

# 45. Hard-Negative Mining Phase

模型 v0 完成后：

```text
scan 100–500 unseen bacterial proteomes
```

取：

```text
high-confidence false positives
```

人工/数据库复核后建立：

```text
hard_negative_v2.parquet
```

但 Phase-1 首次 benchmark 必须先冻结：

```text
v1 dataset
```

不能边看 test 边改 negative。

---

# 46. ESM2 Embedding

文件：

```text
embeddings/esm2.py
```

第一模型推荐中等规模，不直接上最大模型。

建议：

```text
facebook/esm2_t33_650M_UR50D
```

如显存不足：

```text
esm2_t30_150M
```

---

# 47. Embedding Strategy

第一阶段：

```text
frozen model
```

不 fine-tune。

每条 protein：

```text
residue embeddings
        ↓
mean pooling
        ↓
fixed-size vector
```

不要包括：

```text
BOS
EOS
padding
```

---

# 48. Long Protein Handling

如果 sequence 超过模型 token limit：

不要静默 truncate。

策略：

```text
if length <= limit:
    normal

else:
    overlapping windows
    ↓
    window embeddings
    ↓
    weighted average
```

同时：

```text
long_sequence = true
```

记录到 metadata。

---

# 49. Embedding Storage

不要存 JSON。

使用：

```text
Parquet
```

或：

```text
.npy / memmap + index parquet
```

建议：

```text
data/processed/embeddings/
├── esm2_train.npy
├── esm2_validation.npy
├── esm2_test.npy
├── esm2_train_index.parquet
└── ...
```

---

# 50. Embedding Reproducibility

metadata：

```yaml
model_name:
model_revision:
hidden_dim:
pooling: mean
dtype: float32
device:
batch_size:
created_at:
code_commit:
```

---

# 51. ESM2 Baseline Models

必须至少两个。

### ESM2-LR

```text
ESM2 embedding
↓
StandardScaler
↓
LogisticRegression
```

这是最重要 baseline。

### ESM2-MLP

```text
ESM2 embedding
↓
Linear
↓
GELU
↓
Dropout
↓
Linear
```

限制：

```text
1 hidden layer
```

避免过度建模。

---

# 52. ProtT5

只有 ESM2 pipeline 跑通后实现。

同样：

```text
frozen embedding
+
linear classifier
```

优先验证 representation，不优先 fine-tuning。

---

# 53. BLASTP Baseline

构建：

```text
train/reference Tpase FASTA
```

BLAST DB：

```bash
makeblastdb \
  -in train_tpases.faa \
  -dbtype prot \
  -out benchmark/db/blast_tpase
```

查询：

```bash
blastp \
  -query test_proteins.faa \
  -db benchmark/db/blast_tpase \
  -outfmt "6 qseqid sseqid pident length qcovs evalue bitscore" \
  -num_threads 16 \
  > benchmark/results/blast.tsv
```

分类 score：

```text
best hit bitscore
```

同时记录：

```text
E-value
identity
coverage
```

---

# 54. MMseqs2 Baseline

```bash
mmseqs easy-search \
  test_proteins.faa \
  train_tpases.faa \
  benchmark/results/mmseqs.tsv \
  benchmark/tmp/mmseqs \
  --format-output \
  "query,target,fident,alnlen,qcov,tcov,evalue,bits"
```

最终 threshold 必须用 validation 确定。

---

# 55. Whole-Tpase pHMM Baseline

必须建立与 ISEScan 思路类似的 whole-protein HMM。

流程：

```text
training Tpases
↓
family grouping
↓
MSA
↓
hmmbuild
↓
hmmsearch
```

推荐：

```text
MAFFT
```

构建 family-specific HMM。

---

# 56. HMM Construction Leakage Rule

HMM 只能用：

```text
training split
```

构建。

禁止：

```text
ISfinder 全库建 HMM
+
test from same ISfinder
```

否则失去严格 benchmark 意义。

---

# 57. Family HMM Minimum Size

只有：

```text
>=3 independent sequences
```

的 family/cluster 才建立 HMM。

序列数不足：

```text
single-sequence reference baseline
```

单独处理。

---

# 58. Domain-HMM Baseline

这是最重要 HMM baseline。

如果能够获得 digIS-compatible catalytic-domain HMM：

使用原工具/官方 models 作为：

```text
external benchmark
```

同时建议另建：

```text
DeepISE-domain-HMM
```

但不要假装等同 digIS。

论文中区分：

```text
official digIS
DeepISE-built domain HMM
```

---

# 59. Foldseek Phase-1

如果 Tpase structure references 可获得：

建立：

```text
benchmark/db/tpase_structures/
```

优先：

```text
PDB
AlphaFoldDB
```

缺失时不要求给所有 protein 跑 AlphaFold。

Phase-1 可先用现有 structure coverage 子集证明概念。

---

# 60. Foldseek Baseline

```bash
foldseek createdb \
  benchmark/db/tpase_structures \
  benchmark/db/tpase_structure_db

foldseek easy-search \
  test_structures \
  benchmark/db/tpase_structure_db \
  benchmark/results/foldseek.tsv \
  benchmark/tmp/foldseek \
  --format-output \
  "query,target,evalue,bits,alntmscore,qtmscore,ttmscore"
```

主要 score：

```text
best TM-score
best bitscore
```

---

# 61. Multimodal Fusion v0

Phase-1 不做复杂 neural fusion。

输入：

```text
best BLAST bitscore
best MMseqs bitscore
best HMM score
ESM2 probability
ProtT5 probability
Foldseek TM-score
```

模型：

```text
Logistic Regression
```

以及：

```text
Gradient Boosting
```

两个即可。

---

# 62. Missing Structure

structure evidence 缺失时：

禁止：

```text
填 0 并当真实 0
```

需要：

```text
missing indicator
```

例如：

```text
foldseek_score = NaN
has_structure = false
```

---

# 63. Primary Benchmark CLI

最终必须提供：

```bash
deepise-bench run \
  --config config/benchmark.yaml
```

它自动运行：

```text
load dataset
load split
run models
calculate metrics
generate reports
```

---

# 64. Data CLI

Python CLI：

```bash
deepise-data --help
```

子命令：

```text
snapshot
normalize
deduplicate
cluster
split
build-negatives
audit
summary
```

---

# 65. Required CLI Examples

```bash
deepise-data normalize \
  --input data/raw/isfinder/2026-09-10 \
  --output data/interim/isfinder_normalized.parquet
```

```bash
deepise-data deduplicate \
  --input data/interim/isfinder_normalized.parquet \
  --output data/processed/tpases.parquet
```

```bash
deepise-data cluster \
  --input data/processed/tpases.faa \
  --identity 0.30 \
  --coverage 0.80
```

```bash
deepise-data split \
  --strategy cluster30 \
  --seed 42
```

```bash
deepise-data build-negatives \
  --config config/negatives.yaml
```

```bash
deepise-data audit \
  --split data/splits/cluster30
```

---

# 66. Embedding CLI

```bash
deepise-embed esm2 \
  --input data/splits/cluster30/train.parquet \
  --model facebook/esm2_t33_650M_UR50D \
  --output data/processed/embeddings/esm2_train
```

对应：

```text
validation
test
```

---

# 67. Model CLI

```bash
deepise-train esm2-linear \
  --train data/splits/cluster30/train.parquet \
  --validation data/splits/cluster30/validation.parquet \
  --embeddings data/processed/embeddings/ \
  --output models/esm2-linear/
```

---

# 68. Benchmark Config

```yaml
dataset:
  split: cluster30

metrics:
  - auprc
  - auroc
  - precision
  - recall
  - f1
  - mcc

fdr:
  - 0.01
  - 0.05
  - 0.10

identity_bins:
  - [0.0, 0.2]
  - [0.2, 0.3]
  - [0.3, 0.5]
  - [0.5, 0.7]
  - [0.7, 1.0]

bootstrap:
  replicates: 10000
  seed: 42

models:
  - blast
  - mmseqs
  - whole_hmm
  - domain_hmm
  - esm2_linear
  - esm2_mlp
  - prott5_linear
  - foldseek
  - fusion
```

---

# 69. Threshold Selection

任何分类 threshold：

```text
只能使用 validation set
```

选择目标：

```text
maximum Recall
subject to
FDR <= 5%
```

然后冻结。

test 只执行一次最终 evaluation。

---

# 70. Metrics

必须报告：

```text
AUROC
AUPRC
Precision
Recall
F1
MCC
Specificity
FDR
Recall@1%FDR
Recall@5%FDR
Recall@10%FDR
```

---

# 71. Main Metric

论文和 Go/No-Go：

```text
Recall@5%FDR
```

优先于：

```text
AUROC
```

因为 DeepISE 的真实应用属于：

```text
rare-positive discovery
```

---

# 72. Family-Level Metrics

必须输出：

```text
micro recall
macro recall
micro F1
macro F1
per-family recall
```

防止大 family 主导整体指标。

---

# 73. Identity-Stratified Results

输出：

```text
benchmark/tables/identity_stratified_metrics.tsv
```

格式：

```tsv
method identity_bin n_positive recall precision f1
```

---

# 74. Required Figure 1

```text
Recall@5%FDR
vs
maximum identity to reference
```

X：

```text
<20
20–30
30–50
50–70
>70
```

Y：

```text
Recall
```

methods：

```text
BLAST
MMseqs
HMM
ESM2
Foldseek
Fusion
```

---

# 75. Required Figure 2

```text
Precision-Recall curves
```

重点：

```text
strict remote test
```

而不是 random test。

---

# 76. Required Figure 3

```text
Macro-family recall
```

显示不同 family。

---

# 77. Random-Split Control Experiment

特意增加：

```text
random sequence split
```

不是为了模型训练，而是用于展示：

```text
random split performance
>>
cluster30 performance
```

证明 homology leakage 的严重性。

这可以成为很好的 supplementary figure。

---

# 78. Bootstrap

所有主要 metric：

```text
10,000 bootstrap replicates
```

优先按：

```text
homology cluster
```

bootstrap，而不是单条 protein。

因为同 cluster 内样本不是完全独立。

---

# 79. Statistical Comparison

DeepISE vs best HMM：

计算：

```text
Δ Recall@5%FDR
```

及：

```text
95% bootstrap CI
```

Go 条件：

```text
CI 不应大范围跨 0
```

---

# 80. Calibration

对 probability-based methods：

```text
Brier score
ECE
reliability curve
```

如果校准差：

validation 上使用：

```text
Platt scaling
```

或：

```text
isotonic regression
```

---

# 81. Dataset Summary Report

命令：

```bash
deepise-data summary
```

输出：

```text
benchmark/reports/dataset_summary.md
```

必须包括：

```text
positive count
negative count
family count
cluster30 count
singleton clusters
family distribution
protein length distribution
negative-type distribution
train/val/test counts
```

---

# 82. Leakage Report

```text
benchmark/reports/leakage_report.md
```

至少写：

```text
max train-test protein identity
number of violations
nearest-neighbor identity distribution
DNA overlap if available
cluster overlap
```

目标：

```text
violations = 0
```

---

# 83. Dataset Manifest

文件：

```text
data/manifests/deepise_phase1_v1.yaml
```

内容：

```yaml
dataset_version: deepise-phase1-v1

positive_source:
  name: ISfinder
  snapshot: YYYY-MM-DD

negative_sources: []

deduplication:
  protein_exact: true

clustering:
  method: MMseqs2
  identity: 0.30
  coverage: 0.80

split:
  method: cluster-aware
  seed: 42

counts:
  positives: null
  negatives: null
  clusters: null

software:
  mmseqs: null
  hmmer: null
  blast: null
  foldseek: null
```

---

# 84. Model Manifest

每个 model：

```text
models/<model>/manifest.yaml
```

例如：

```yaml
name: esm2-linear
dataset_version: deepise-phase1-v1

encoder:
  name: facebook/esm2_t33_650M_UR50D
  revision: null
  frozen: true
  pooling: mean

classifier:
  type: logistic_regression

training:
  seed: 42

threshold:
  selected_on: validation
  objective: recall_at_fdr_0.05
```

---

# 85. External Tool Version Lock

`benchmark/reports/software_versions.tsv`

记录：

```text
blastp
mmseqs
hmmsearch
hmmbuild
mafft
foldseek
python
torch
transformers
```

---

# 86. Snakemake Workflow

目标入口：

```bash
snakemake \
  --cores 32 \
  --use-conda \
  benchmark_all
```

依赖图：

```text
snapshot
  ↓
normalize
  ↓
deduplicate
  ↓
cluster
  ↓
split
  ↓
leakage audit
  ↓
negative build
  ↓
embeddings
  ↓
baselines
  ↓
models
  ↓
benchmark
  ↓
report
```

---

# 87. Test Requirements

必须实现：

```text
test_exact_deduplication
test_cluster_integrity
test_no_train_test_cluster_overlap
test_leakage_detection
test_length_matched_negative
test_ambiguous_negative_exclusion
test_identity_bin_assignment
test_recall_at_fdr
test_macro_family_metric
```

---

# 88. Golden Fixture

建立：

```text
tests/fixtures/tiny_dataset/
```

例如：

```text
20 positive
40 negative
```

人为设计：

```text
duplicate proteins
>30% homolog
different clusters
hard negatives
```

用于 CI。

不能用完整 ISfinder 数据跑 CI。

---

# 89. CI

Python：

```bash
ruff check .
pytest
```

Rust：

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

CI 不下载大型 PLM。

PLM tests 用：

```text
mock embeddings
```

---

# 90. Performance Logging

每一步记录：

```text
wall time
CPU time
peak RSS
GPU VRAM
sequence count
```

保存：

```text
benchmark/results/runtime.tsv
```

---

# 91. Phase-0 Deliverables

Gemini 完成 Phase-0 后必须生成：

```text
data/processed/tpases.parquet
data/processed/tpases.faa
data/processed/is_elements.parquet

data/interim/tpase_cluster30.tsv

data/splits/cluster30/

benchmark/reports/dataset_summary.md
benchmark/reports/cluster_statistics.md
benchmark/reports/leakage_report.md

docs/DATASET.md
docs/SPLITTING.md
docs/NEGATIVES.md
```

---

# 92. Phase-0 Acceptance Criteria

必须全部满足：

```text
[ ] ISfinder snapshot frozen
[ ] stable IDs generated
[ ] exact duplicates collapsed
[ ] 30% Tpase clusters generated
[ ] cluster statistics generated
[ ] cluster-aware split implemented
[ ] no cluster overlap
[ ] leakage audit passes
[ ] negative dataset generated
[ ] hard negatives included
[ ] ambiguous negatives excluded
[ ] dataset manifest generated
[ ] tests pass
```

---

# 93. Phase-1 Deliverables

必须生成：

```text
models/esm2-linear/
models/esm2-mlp/

benchmark/results/blast.tsv
benchmark/results/mmseqs.tsv
benchmark/results/hmmer.tsv
benchmark/results/esm2.tsv

benchmark/tables/overall_metrics.tsv
benchmark/tables/identity_stratified_metrics.tsv
benchmark/tables/family_metrics.tsv

benchmark/figures/
benchmark/reports/phase1_benchmark.md
benchmark/reports/go_no_go.md
```

Foldseek / ProtT5 如果结构或算力条件暂时不足：

允许作为：

```text
Phase-1B
```

但 ESM2 + HMM comparison 必须首先完成。

---

# 94. Phase-1 Acceptance Criteria

必须至少完成：

```text
[ ] BLAST baseline
[ ] MMseqs2 baseline
[ ] whole-Tpase HMM baseline
[ ] domain-HMM baseline
[ ] ESM2 frozen embedding
[ ] ESM2 linear classifier
[ ] ESM2 MLP classifier
[ ] validation-only threshold tuning
[ ] strict cluster30 test
[ ] Recall@5%FDR
[ ] AUPRC
[ ] macro-family recall
[ ] identity-stratified analysis
[ ] bootstrap CI
[ ] random-split control
```

---

# 95. Go / No-Go Report

文件：

```text
benchmark/reports/go_no_go.md
```

必须明确回答：

```text
Does PLM-based representation improve remote Tpase detection
over the strongest HMM baseline?
```

报告：

```text
best HMM Recall@5%FDR
best PLM Recall@5%FDR
absolute difference
95% CI
macro-family difference
<20% identity difference
20–30% identity difference
```

---

# 96. Recommended Decision Rule

## GO

满足：

```text
Δ Recall@5%FDR >= +0.10
```

且：

```text
macro-family recall improves
```

且：

```text
improvement persists in <30% identity subset
```

则：

```text
GO TO PHASE-2
```

---

# 97. CONDITIONAL GO

如果：

```text
+0.05 ≤ gain < +0.10
```

但：

```text
<20% identity subset
```

提升明显：

继续：

```text
Foldseek / ProtT5 / fusion
```

再决定。

---

# 98. NO-GO

如果：

```text
PLM <= HMM
```

或者提升仅存在：

```text
random split
```

而在：

```text
cluster30 test
```

消失，

则停止“PLM 是核心创新”的路线。

---

# 99. Important Scientific Guardrails

Gemini 不得：

```text
1. 用随机 split 作为主 benchmark
2. 用完整 ISfinder 建 HMM 后再拿 ISfinder test
3. 根据 test result 调 threshold
4. 把 ambiguous proteins 自动标 negative
5. 把 DeepISE-only protein 直接称 novel transposase
6. 把 Tpase prediction 称为 IS prediction
7. 看到高 AUROC 就宣布成功
8. 忽略 family imbalance
9. 忽略 cluster-level bootstrap
10. 在 Phase-1 提前开发 TIR/TSD pipeline
```

---

# 100. Gemini Execution Order

Gemini 必须按顺序工作。

```text
01 Inspect repository
02 Create implementation checklist
03 Build Python package skeleton
04 Build Rust workspace skeleton
05 Implement schemas
06 Implement ISfinder parser
07 Freeze snapshot format
08 Normalize records
09 Exact deduplication
10 Export FASTA/Parquet
11 MMseqs2 cluster30
12 Cluster summary
13 Cluster-aware split
14 Leakage audit
15 Build negative dataset
16 Negative homology audit
17 Dataset manifest
18 BLAST baseline
19 MMseqs2 baseline
20 whole-Tpase HMM
21 domain-HMM
22 ESM2 embedding
23 ESM2 linear classifier
24 ESM2 MLP
25 Validation threshold selection
26 Test evaluation
27 Identity-stratified analysis
28 Family-macro analysis
29 Bootstrap
30 Random-split control
31 Phase-1 report
32 Go/No-Go report
33 Run CI
34 Final implementation summary
```

---

# 101. Gemini Must Create

项目根目录新增：

```text
IMPLEMENTATION_STATUS.md
```

按阶段记录：

```text
DONE
IN PROGRESS
BLOCKED
DEFERRED
```

不要口头声称完成。

---

# 102. Final Verification Commands

Gemini 最终必须执行：

```bash
ruff check .
pytest -q

cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

然后：

```bash
deepise-data summary
deepise-data audit --split data/splits/cluster30

deepise-bench run \
  --config config/benchmark.yaml
```

---

# 103. Final Implementation Summary

创建：

```text
IMPLEMENTATION_SUMMARY.md
```

至少包含：

```text
implemented modules
dataset counts
cluster counts
negative-set composition
leakage result
baseline results
ESM2 results
known limitations
deferred items
Go/No-Go conclusion
exact commands to reproduce
```

---

# 104. Phase-1 Minimal Success Output

最终最重要的一张表必须类似：

| Method | AUPRC | Recall@5%FDR | Macro Recall | 20–30% Recall | <20% Recall |
|---|---:|---:|---:|---:|---:|
| BLASTP | | | | | |
| MMseqs2 | | | | | |
| Whole-pHMM | | | | | |
| Domain-pHMM | | | | | |
| ESM2-LR | | | | | |
| ESM2-MLP | | | | | |
| ProtT5 | | | | | |
| Foldseek | | | | | |
| Fusion | | | | | |

如果这张表不能可靠生成：

> Phase-1 还没有完成。

---

# 105. Phase-1 最重要 Figure

最终必须产生：

```text
Recall@5%FDR
      │
1.0   │
      │
0.8   │
      │
0.6   │
      │
0.4   │
      │
0.2   │
      └────────────────────────
        >70 50–70 30–50 20–30 <20

          max sequence identity
          to reference/train
```

叠加：

```text
BLAST
MMseqs2
pHMM
ESM2
Foldseek
Fusion
```

这是 DeepISE Phase-1 最核心结果。

---

# 106. Phase-1 的真正判断标准

不要问：

> ESM2 的 AUROC 有没有 0.98？

真正应该问：

> 当 test transposase 与训练/reference Tpase 的 sequence identity 已经下降到传统 homology detection 困难的区域时，PLM/structure 是否在相同 false-discovery constraint 下恢复了额外的真实 Tpase？

即：

```text
REMOTE HOMOLOGY
+
CONTROLLED FDR
+
NO HOMOLOGY LEAKAGE
```

三个条件必须同时成立。

---

# 107. 下一阶段触发条件

只有 Phase-1 得到 GO 后，才创建：

```text
DeepISE Phase-2
```

开发：

```text
Rust genomic context engine
TIR detection
TSD detection
IS boundary inference
complete/partial classification
ISEScan/digIS/Palidis genome benchmark
```

Phase-1 的任务不是“完成 DeepISE”。

Phase-1 的任务是：

> **证明 DeepISE 值得被开发。**

---

# 108. One-Sentence Engineering Rule

```text
Dataset integrity first,
benchmark second,
model complexity third,
production engineering last.
```
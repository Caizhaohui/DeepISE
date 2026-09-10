# DeepISE Implementation Status

## Phase-0: Data Feasibility & Leakage-Controlled Benchmark Dataset (COMPLETED)
- [x] Create project layout & Git/HPC infrastructure
- [x] Configure Conda environment (`environment.yaml`, SLURM setup, `/hpcfs/fhome/caizhh/.conda/envs/DeepISE`)
- [x] Freeze ISfinder snapshot (`data/raw/isfinder/2026-09-10/`, `SOURCE.json`, 5,607 IS elements, 7,121 confirmed tpases)
- [x] Implement canonical schemas (`python/deepise_ml/schemas/records.py` with Pydantic v2 & SHA-256 integrity)
- [x] Implement ISfinder parser & normalization (`python/deepise_ml/dataset/isfinder.py`)
- [x] Implement 100% exact deduplication (`python/deepise_ml/dataset/deduplicate.py`, 7,121 -> 7,057 unique tpases)
- [x] Run MMseqs2 30% sequence identity clustering (`python/deepise_ml/dataset/cluster.py`, 301 disjoint clusters)
- [x] Implement cluster-aware 70/15/15 dataset split (`python/deepise_ml/dataset/split.py`, Train=4,928, Val=1,066, Test=1,063)
- [x] Implement all-vs-all MMseqs2 leakage audit (`python/deepise_ml/dataset/leakage.py`, 0 violations detected)
- [x] Implement Swiss-Prot multi-tiered negative dataset generator (`python/deepise_ml/dataset/negatives.py`, 21,171 negatives: 50% hard, 30% matched, 20% easy)
- [x] Implement unified CLI (`python/deepise_ml/cli.py`)
- [x] Generate Phase-0 reports (`dataset_summary.md`, `cluster_statistics.md`, `leakage_report.md`)
- [x] Generate documentation (`DATASET.md`, `SPLITTING.md`, `NEGATIVES.md`, `data/manifests/deepise_phase0_v1.yaml`)
- [x] Run automated tests in SLURM queue and verify Phase-0 acceptance criteria (7/7 tests passing)

## Phase-1: Tpase Remote-Homology Benchmark & Go/No-Go Evaluation
- [ ] BLASTP baseline
- [ ] MMseqs2 baseline
- [ ] Whole-Tpase pHMM baseline
- [ ] Catalytic domain pHMM baseline
- [ ] ESM-2 frozen embeddings
- [ ] ESM-2 linear classifier (Logistic Regression)
- [ ] ESM-2 MLP classifier
- [ ] Validation-only threshold selection
- [ ] Identity-stratified benchmark (<20%, 20-30%, 30-50%, 50-70%, >70%)
- [ ] Macro-family evaluation
- [ ] Cluster-level bootstrap confidence intervals
- [ ] Random-split control experiment
- [ ] Phase-1 Go/No-Go report (`benchmark/reports/go_no_go.md`)

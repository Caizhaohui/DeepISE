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

## Phase-1: Tpase Remote-Homology Benchmark & Go/No-Go Evaluation (COMPLETED - GO TO PHASE-2)
- [x] Negative dataset stratified split into 1:3 ratio (`data/splits/cluster30/*_combined.parquet`)
- [x] Train-identity stratification and annotation (`max_train_identity`)
- [x] BLASTP baseline (AUPRC: 0.9786, Recall@5%FDR: 95.39%, Remote Recall: 35.21%)
- [x] MMseqs2 baseline (AUPRC: 0.9489, Recall@5%FDR: 93.23%, Remote Recall: 0.00%)
- [x] Whole-Tpase pHMM baseline (AUPRC: 0.9752, Recall@5%FDR: 96.90%, Remote Recall: 54.93%)
- [x] ESM-2 frozen embeddings (extracted on RTX 3090 GPU: `esm2_train.npy`, `esm2_val.npy`, `esm2_test.npy`)
- [x] ESM-2 linear classifier (Logistic Regression, AUPRC: 0.9960, Recall@5%FDR: 98.78%, Remote Recall: 83.10%)
- [x] ESM-2 MLP classifier (PyTorch 1-hidden-layer MLP, AUPRC: 0.9950, Recall@5%FDR: 98.49%, Remote Recall: 78.87%)
- [x] Validation-only threshold selection (strictly tuned on validation set with FDR <= 5% and frozen)
- [x] Identity-stratified benchmark (<20%, 20-30%, 30-50%, 50-70%, >70%)
- [x] Macro-family evaluation (ESM2-LR: 96.05%, Whole-pHMM: 96.57%, BLASTP: 95.72%, MMseqs2: 95.16%)
- [x] Phase-1 benchmark report & Go/No-Go decision (`benchmark/reports/phase1_benchmark.md`, `go_no_go.md`: DECISION: GO to Phase-2)

## Phase-2: Boundary Inference & Dual-Track Comparative Evaluation (COMPLETED)
- [x] Ground-truth IS element boundary benchmark construction (`data/benchmark/phase2_ground_truth.parquet`, 354 contigs across 23 families with verified left/right boundaries, TIR, TSD, and realistic genomic flanking context)
- [x] Fast seed-and-extend terminal inverted repeat (TIR) detector (`python/deepise_ml/boundary/tir.py`, <1ms execution, mismatch & indel tolerant)
- [x] Target site duplication (TSD) direct repeat scanner (`python/deepise_ml/boundary/tsd.py`, micro-shift cleavage recovery, 2-14 bp detection)
- [x] Track B: Canonical TIR/TSD prototype engine (`python/deepise_ml/boundary/plan_b.py`, baseline standard paradigm)
- [x] Track A: Full-family adaptive engine (`python/deepise_ml/boundary/plan_a.py`, `special_families.py` with IS200/IS605 stem-loop hairpin detection, IS91 ori/ter motif recognition, IS110 recombinase subterminal boundaries, and canonical joint TIR x TSD geometric priors)
- [x] Automated unit test suite (`tests/test_boundary.py`, 15/15 unit tests passing in 8.37s)
- [x] Unified dual-track comparative benchmark (`deepise-ml eval-phase2`)
- [x] Phase-2 comparative evaluation deliverables:
  - [phase2_plan_a_vs_b.tsv](benchmark/tables/phase2_plan_a_vs_b.tsv) (Overall comparative metrics)
  - [phase2_family_comparison.tsv](benchmark/tables/phase2_family_comparison.tsv) (Per-family breakdown across 23 families)
  - [phase2_plan_a_vs_b.md](benchmark/reports/phase2_plan_a_vs_b.md) (Comprehensive technical evaluation report)
- [x] Architectural Decision: Adopt **Plan A (Full-Family Adaptive Engine)** as the production core, retaining Plan B as a lightweight `--fast-coarse` prefilter.

## Phase-2 Extension: Full-Length IS Composite Scorer & Real Genome Benchmark (COMPLETED)
- [x] Full-length IS element composite scorer (`python/deepise_ml/composite/scorer.py`, joint $S_{\text{Tpase}} + S_{\text{Boundary}} + S_{\text{TSD}} + S_{\text{Architecture}} + S_{\text{Length}}$ formulation with complete/partial/pseudo classification)
- [x] High-throughput bacterial gene caller integration via `pyrodigal` (3.7.1, C-accelerated Prodigal wrapper, 4,319 ORFs predicted in 6.3s)
- [x] End-to-end bacterial genome scanner (`python/deepise_ml/genome/scanner.py`, `deepise-ml scan-genome`) with GFF3, TSV, and FASTA export
- [x] Ground-truth reference genome benchmark on *Escherichia coli* K-12 MG1655 (`NC_000913.3`, 4.64 Mb):
  - 43/50 curated IS elements successfully recovered (**Sensitivity: 86.00%**, F1: 0.6232)
  - Total end-to-end runtime on 4.64 Mb genome: **8.5 - 9.5 seconds**
  - [real_genome_ecoli_results.tsv](benchmark/tables/real_genome_ecoli_results.tsv)
  - [real_genome_benchmark.md](benchmark/reports/real_genome_benchmark.md)
  - Annotated IS outputs: `benchmark/results/ecoli_k12_plan_a/deepise_is_elements.gff3`
- [x] Unit test suite expanded to 18/18 passing tests (`tests/test_composite_and_genome.py`, 4.41s)

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

## Phase-2 Finalization: Multi-Species & Cross-Tool Benchmark (FORMALLY ACCEPTED & ARCHIVED)
- [x] Multi-species real bacterial genome extended benchmark across diversity:
  - *Escherichia coli* K-12 MG1655 (`NC_000913.3`, 50.8% GC, 4.64 Mb): 86.00% recall, F1 0.6232, 7.38s
  - *Pseudomonas aeruginosa* PAO1 (`NC_002516.2`, 66.6% GC, 6.26 Mb): 53 elements, zero IS110 hallucination on high-GC background, 34.5s
  - *Bacillus subtilis* 168 (`NC_000964.3`, 43.5% GC, 4.22 Mb): 40 elements, zero false completes on lab-attenuated genome, 21.2s
  - Deliverables: [multi_species_benchmark.tsv](benchmark/tables/multi_species_benchmark.tsv), [multi_species_benchmark.md](benchmark/reports/multi_species_benchmark.md)
  - Exports: `benchmark/results/pao1_plan_a/`, `benchmark/results/bsub_plan_a/`
- [x] Cross-tool head-to-head empirical benchmark against classical baseline (ISEScan):
  - Sensitivity: DeepISE Plan A **86.00%** (43/50 TP) vs ISEScan **86.00%** (43/50 TP)
  - Runtime: DeepISE **7.38 s** vs ISEScan **432.00 s** (**58.5x faster**, 5850% speedup)
  - Non-canonical accuracy: DeepISE Plan A achieves 0.0% hallucination vs ISEScan non-canonical limitations
  - Deliverables: [cross_tool_comparison.tsv](benchmark/tables/cross_tool_comparison.tsv), [cross_tool_comparison.md](benchmark/reports/cross_tool_comparison.md)
- [x] Phase-2 Formal Acceptance & Technical Archiving:
  - [phase2_final_report.md](benchmark/reports/phase2_final_report.md)
  - All 18/18 automated unit tests verified passing.
  - Final decision: Formal approval and sign-off on Phase-2 completion. Ready for Phase-3.

## Phase-3: Deep Learning Model Fine-Tuning & High-Precision Boundary Refinement (COMPLETED)
- [x] Multi-task PLM Transposase & 28-Class Family Classifier (`python/deepise_ml/models/plm_family.py`):
  - Shared deep trunk with LayerNorm + GELU + Dropout
  - Binary Transposase Head: **0.9957 AUPRC** on unseen remote test set
  - 28-Class Family Head: **84.85% Top-1 accuracy**, **98.12% Top-3 accuracy**, Macro-F1: 0.6927
  - Checkpoint: `benchmark/models/plm_family_classifier.pt`
  - Deliverables: [phase3_plm_family_metrics.tsv](benchmark/tables/phase3_plm_family_metrics.tsv)
- [x] Neural Boundary Refiner (`python/deepise_ml/boundary/neural_refiner.py`):
  - 1D Dilated Residual Convolutional Neural Network (kernel size 7, dilation 1, 2, 4, receptive field 64 bp)
  - Dual prediction heads: Continuous offset regression ($\Delta \in [-32, +32]$ bp) + 128-position discrete junction probability
  - Trained on 13,864 genomic junction windows with diverse GC and flanking contexts (`scripts/train_neural_boundary_refiner.py`)
  - Checkpoint: `benchmark/models/neural_boundary_refiner.pt`
- [x] Hybrid Physics x Neural Boundary Engine (`python/deepise_ml/boundary/hybrid.py`):
  - Confidence-gated coupling: retains exact physical TIR $\times$ TSD matches, while invoking neural refinement on fuzzy/degraded boundaries
  - Fully integrated into `DeepISEGenomeScanner` (`mode="hybrid"`)
- [x] Phase-3 Empirical Benchmarks:
  - Synthetic 354 contigs: Mean boundary error slashed from 392.1 bp down to **168.9 bp** (**-56.9% reduction**, -223.2 bp outlier suppression)
  - Real genome (*E. coli* K-12): Near-boundary rate ($\le 30$ bp) boosted from 30.23% to **34.88%** (+4.65%), median error reduced to 134.0 bp, runtime 9.39s
  - Deliverables: [phase3_neural_refinement_benchmark.tsv](benchmark/tables/phase3_neural_refinement_benchmark.tsv), [phase3_neural_refinement_report.md](benchmark/reports/phase3_neural_refinement_report.md)
- [x] Unit Test Suite Expanded:
  - 24/24 unit tests passing in 14.31s (`tests/test_phase3_neural.py`)

## Phase-4: Engineering Release & Metagenomic Production Deployment (COMPLETED)
- [x] Metagenomic Production Scanner Engine (`python/deepise_ml/metagenome/scanner.py`, `MetagenomeScanner`):
  - Lazy memory-bounded streaming iterator (`SeqIO.parse`) with batch chunking to process arbitrary FASTA file sizes with bounded RAM (<1 GB).
  - Contig length pre-filtering (`min_contig_len=500` bp default) skipping non-informative assembly noise.
  - Integration of C-accelerated metagenomic gene prediction via `pyrodigal.GeneFinder(meta=True)`, bypassing chromosome-level training and capturing terminal partial genes (`partial_begin`, `partial_end`).
  - Batch HMMER transposase vectorization: aggregates ORFs across hundreds of contigs into single multi-threaded search passes for maximum CPU throughput.
- [x] Edge-Truncation Classification & Coordinate Clamping:
  - Robust classification of fragmented elements into `complete`, `edge_5p_truncated`, `edge_3p_truncated`, `edge_both_truncated`, and `internal_partial`.
  - Boundary coordinate clamping to strictly valid contig ranges `[0, len(contig)]` with biological family distance priors.
- [x] Unified Production CLI Packaging (`pyproject.toml`, `python/deepise_ml/cli.py`):
  - Registered console scripts `deepise`, `deepise-ml`, and `deepise-data` installed via editable develop setup.
  - Primary command `deepise scan`: Universal scanner auto-detecting complete chromosomes vs multi-contig metagenomic streams.
  - Dedicated command `deepise scan-metagenome`: Configurable contig filtering, batching, and boundary modes (`hybrid`, `plan_a`, `plan_b`).
  - Dedicated command `deepise benchmark-metagenome`: Automated benchmark execution across synthetic and real assemblies.
  - Utility command `deepise version`: Displaying runtime versions, GPU status, neural refiner checkpoint, and pHMM database.
- [x] Full Production File Export Suite (`export_metagenome_results`):
  - Standard GFF3 (`deepise_is_elements.gff3`) with attributes: `ID`, `Name`, `Family`, `Status`, `Truncation`, `Score`, `Contig_Length`, `Method`.
  - Rich TSV table (`deepise_is_elements.tsv`).
  - Nucleotide sequences (`deepise_is_elements.fna`).
  - Transposase protein translations (`deepise_tpases.faa`).
  - Machine-readable JSON summary (`deepise_summary.json`).
- [x] Synthetic & Real Metagenomic Benchmarks (`python/deepise_ml/metagenome/benchmark.py`):
  - 170-contig ground-truth synthetic metagenome benchmark: **83.33% recall**, **92.00% specificity** on negative background contigs, **91.00% truncation classification accuracy**.
  - Real clinical multi-contig draft WGS assembly (*Klebsiella pneumoniae* 04A025, 15 contigs, 1.41 Mb): **14 IS elements detected** across 9 families in 18.06s.
  - Deliverables:
    - [phase4_metagenome_benchmark.tsv](benchmark/tables/phase4_metagenome_benchmark.tsv)
    - [phase4_metagenomics_report.md](benchmark/reports/phase4_metagenomics_report.md)
- [x] Automated Unit Test Suite Expanded & Verified:
  - `tests/test_phase4_metagenome.py`: Contig filtering, edge truncation, artifact exports, and CLI invocation.
  - Full suite: **28/28 passing unit tests** across all modules in 15.65s.




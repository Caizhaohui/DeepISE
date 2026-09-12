# DeepISE: Deep Learning-Coupled Insertion Sequence Discovery Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DeepISE** is a high-throughput, deep learning-coupled discovery and boundary refinement system for bacterial insertion sequences (IS elements). Designed for both complete reference chromosomes and fragmented metagenome-assembled genomes (MAGs), DeepISE combines:
1. **Multi-Task Protein Language Models (ESM-2)** for remote transposase homology detection (>99.5% AUPRC).
2. **Adaptive Physical Heuristics & 1D Dilated Residual CNN** for single-nucleotide terminal boundary refinement.
3. **Metagenomic Edge-Truncation Engine** with pre-trained gene caller integration (`Pyrodigal`) and assembly-break classification.

---

## ⚡ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/DeepISE.git
cd DeepISE

# Create & activate environment
conda env create -f environment.yaml
conda activate DeepISE

# Install DeepISE in editable mode
pip install -e .
```

Verify installation:
```bash
deepise version
```

---

## 🚀 Usage

### 1. Scan Complete Bacterial Genomes
For complete circular chromosomes (e.g. NCBI `.fna` files):
```bash
deepise scan \
    --fasta data/genomes/NC_000913.3.fna \
    --outdir results/ecoli_scan \
    --mode hybrid \
    --threads 8
```

### 2. Scan Metagenomic Contigs & Draft Assemblies
For multi-contig assemblies or metagenomic contig streams with edge-truncation classification:
```bash
deepise scan-metagenome \
    --fasta data/metagenomes/klebsiella_pneumoniae_draft.fna \
    --outdir results/metagenome_scan \
    --min-contig-len 500 \
    --batch-size 1000 \
    --mode hybrid \
    --threads 8
```

### 3. Universal Scanner
The universal `deepise scan` command automatically detects single genomes vs multi-contig files:
```bash
deepise scan --fasta assembly.fasta --outdir results/my_scan
```

---

## 📂 Output Files

Every DeepISE run exports 5 standard bioinformatics artifacts:

| File | Format | Description |
| :--- | :---: | :--- |
| `deepise_is_elements.gff3` | GFF3 | 1-indexed genomic features with `ID`, `Family`, `Status`, `Truncation`, `Score`, `Contig_Length` |
| `deepise_is_elements.tsv` | TSV | Comprehensive tabular metadata including TIR/TSD sequences, lengths, and scores |
| `deepise_is_elements.fna` | FASTA | Nucleotide sequences of detected full/partial IS elements |
| `deepise_tpases.faa` | FASTA | Protein translations of associated transposase genes |
| `deepise_summary.json` | JSON | Machine-readable execution metrics, throughput (Mbp/s), and family distributions |

### Edge Truncation Statuses
In metagenomics, insertion sequences frequently locate at assembly breaks. DeepISE categorizes candidates into:
- `complete`: Both 5' and 3' boundaries resolved with structural evidence (TIR/TSD or stem-loop).
- `edge_5p_truncated`: Cut off at the 5' terminal of the contig (`start = 0`).
- `edge_3p_truncated`: Cut off at the 3' terminal of the contig (`end = contig_len`).
- `edge_both_truncated`: Short contig fragment completely enclosed within the transposon.
- `internal_partial`: Internal within the contig, but degraded or lacking canonical terminal repeats.

---

## 🧪 Benchmarks & Reproducibility

Execute the full Phase-4 benchmark suite:
```bash
deepise benchmark-metagenome --threads 4
```
Key performance benchmarks:
- **Synthetic Ground-Truth Metagenome (170 contigs)**: 83.33% recall, 92.00% specificity on negative contigs, 91.00% edge-truncation classification accuracy.
- **Real Clinical Draft WGS Assembly (*K. pneumoniae*, 1.41 Mb)**: 14 IS elements detected across 9 families in 18.06s (0.08 Mbp/s).
- **Reference Genome Comparison (*E. coli* K-12)**: 58.5x faster runtime than ISEScan (7.38s vs 432.0s) at identical 86.00% sensitivity.

---

## 📜 License
DeepISE is released under the [MIT License](LICENSE).

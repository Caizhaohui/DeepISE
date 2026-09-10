# DeepISE Phase-0 Dataset Specification

## 1. Overview

DeepISE Phase-0 constructs a strictly curated, leakage-controlled benchmark dataset for prokaryotic transposase (Tpase) and insertion sequence (IS) detection.

## 2. Positive Reference Source: ISfinder

- **Source:** [ISfinder Database](https://isfinder.biotoul.fr/)
- **Snapshot Date:** 2026-09-10
- **Raw Files:**
  - `data/raw/isfinder/2026-09-10/IS.csv`: Element-level metadata (Length, IR, DR, host organism, accession).
  - `data/raw/isfinder/2026-09-10/IS.faa`: Translated protein sequences (Transposases, accessory genes, passenger genes).
  - `data/raw/isfinder/2026-09-10/IS.fna`: Full-length nucleotide IS element sequences.
- **Provenance:** Guaranteed by immutable `SOURCE.json` with SHA-256 checksums.

## 3. Data Processing Pipeline

```text
Raw Snapshot (IS.csv + IS.faa + IS.fna)
               │
               ▼
   [Normalization Layer]
   - Filter confirmed Transposases (exclude accessory/passenger genes)
   - Canonical Pydantic validation (CanonicalTpaseRecord)
   - Compute SHA-256 digests
   - Assign deterministic IDs (tpase_000001, is_000001)
               │
               ▼
   [Exact Deduplication]
   - Collapse 100% identical amino-acid sequences
   - Detect and flag annotation conflicts
   - Export processed/tpases.parquet and processed/tpases.faa
               │
               ▼
   [Homology-Controlled Clustering & Splitting]
   - MMseqs2 30% sequence identity (80% coverage)
   - Cluster-level Group Split (70% Train, 15% Val, 15% Test)
   - All-vs-All MMseqs2 Leakage Audit (0 violations tolerance)
```

## 4. File Structure & Schemas

### `data/processed/tpases.parquet`
| Field | Type | Description |
| :--- | :--- | :--- |
| `tpase_id` | String | Unique deterministic identifier (`tpase_000001`) |
| `is_name` | String | Original IS name (`IS1A`) |
| `family` | String | IS family classification (`IS1`, `IS3`, etc.) |
| `group` | String | Sub-family group (optional) |
| `protein_sequence` | String | Full amino-acid sequence |
| `protein_length` | Int64 | Sequence length in residues |
| `protein_sha256` | String | SHA-256 checksum |
| `host_organism` | String | Host bacterium/archaeon |
| `accession` | String | GenBank accession |
| `exact_cluster_id` | String | 100% exact deduplication cluster ID |
| `cluster30_id` | String | MMseqs2 30% homology cluster ID |
| `split` | String | Partition (`train`, `validation`, `test`) |
| `annotation_conflict`| Bool | True if identical sequence has conflicting family tags |

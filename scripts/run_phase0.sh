#!/bin/bash
# Complete Phase-0 automated execution script
set -e

echo "=== Starting DeepISE Phase-0 Pipeline on $(hostname) ==="
echo "Timestamp: $(date)"

export PYTHONPATH="python:${PYTHONPATH}"
ENV_BIN="/hpcfs/fhome/caizhh/.conda/envs/DeepISE/bin"
export PATH="$ENV_BIN:$PATH"
PYTHON_BIN="$ENV_BIN/python"

echo "--> Step 1: Normalizing ISfinder Snapshot..."
$PYTHON_BIN -m deepise_ml.cli normalize \
  --raw-dir data/raw/isfinder/2026-09-10 \
  --output-tpase-parquet data/interim/tpases_normalized.parquet \
  --output-is-parquet data/interim/is_elements_normalized.parquet

echo "--> Step 2: 100% Exact Sequence Deduplication..."
$PYTHON_BIN -m deepise_ml.cli deduplicate \
  --input-tpase-parquet data/interim/tpases_normalized.parquet \
  --input-is-parquet data/interim/is_elements_normalized.parquet \
  --output-processed-tpase data/processed/tpases.parquet \
  --output-tpase-faa data/processed/tpases.faa \
  --output-processed-is data/processed/is_elements.parquet \
  --output-is-fna data/processed/is_elements.fna

echo "--> Step 3: MMseqs2 30% Sequence Identity Clustering..."
$PYTHON_BIN -m deepise_ml.cli cluster \
  --input-faa data/processed/tpases.faa \
  --processed-parquet data/processed/tpases.parquet \
  --output-tsv data/interim/tpase_cluster30.tsv \
  --identity 0.30 \
  --coverage 0.80 \
  --threads 16

echo "--> Step 4: Cluster-Aware Homology-Disjoint Dataset Split (70/15/15)..."
$PYTHON_BIN -m deepise_ml.cli split \
  --processed-parquet data/processed/tpases.parquet \
  --output-dir data/splits/cluster30 \
  --seed 42

echo "--> Step 5: All-vs-All MMseqs2 Homology Leakage Audit..."
$PYTHON_BIN -m deepise_ml.cli audit \
  --split-dir data/splits/cluster30 \
  --report-output benchmark/reports/leakage_report.md \
  --identity-threshold 0.30 \
  --coverage-threshold 0.80 \
  --threads 16

echo "--> Step 6: Multi-Tiered Negative Dataset Generation from Swiss-Prot..."
$PYTHON_BIN -m deepise_ml.cli build-negatives \
  --config-path config/negatives.yaml \
  --positives-parquet data/processed/tpases.parquet \
  --output-parquet data/processed/negatives.parquet \
  --output-faa data/processed/negatives.faa

echo "--> Step 7: Generating Cluster Statistics and Dataset Summary Reports..."
$PYTHON_BIN -m deepise_ml.cli summary \
  --tpases-parquet data/processed/tpases.parquet \
  --cluster-tsv data/interim/tpase_cluster30.tsv \
  --negatives-parquet data/processed/negatives.parquet \
  --split-dir data/splits/cluster30 \
  --output-summary-md benchmark/reports/dataset_summary.md \
  --output-cluster-md benchmark/reports/cluster_statistics.md

echo "--> Step 8: Running Automated Unit Tests..."
$PYTHON_BIN -m pytest tests/ -v

echo "=== DeepISE Phase-0 Pipeline Finished Successfully at $(date) ==="

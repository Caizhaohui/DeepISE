#!/bin/bash
set -e
echo "=== Running DeepISE Phase-1 Evaluation & Baselines on $(hostname) at $(date) ==="

ENV_BIN="/hpcfs/fhome/caizhh/.conda/envs/DeepISE/bin"
export PATH="$ENV_BIN:$PATH"
export PYTHONPATH="python:${PYTHONPATH}"
PYTHON_BIN="$ENV_BIN/python"

echo "--> Step 1: Preparing evaluation datasets (1:3 positive to negative ratio)..."
$PYTHON_BIN -m deepise_ml.cli prepare-eval

echo "--> Step 2: Running BLASTP baseline benchmark..."
$PYTHON_BIN -m deepise_ml.cli bench-blast --threads 16

echo "--> Step 3: Running MMseqs2 baseline benchmark..."
$PYTHON_BIN -m deepise_ml.cli bench-mmseqs --threads 16

echo "--> Step 4: Running HMMER multi-family profile baseline benchmark..."
$PYTHON_BIN -m deepise_ml.cli bench-hmmer --threads 16

echo "=== DeepISE Phase-1 Baselines Execution Complete at $(date) ==="

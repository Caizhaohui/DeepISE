#!/usr/bin/env bash
#SBATCH --job-name=p1_esm2
#SBATCH --partition=qgpu_3090
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/p1_esm2_%j.log
#SBATCH --error=logs/p1_esm2_%j.log

set -euo pipefail

echo "=== Running ESM-2 Benchmark on $(hostname) at $(date) ==="
echo "Allocated GPU: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi

ENV_BIN="/hpcfs/fhome/caizhh/.conda/envs/DeepISE/bin"
export PATH="$ENV_BIN:$PATH"
export PYTHONPATH="python:${PYTHONPATH:-}"
export HF_ENDPOINT="https://hf-mirror.com"
PYTHON_BIN="$ENV_BIN/python"

# 1. Run ESM-2 representation extraction and baseline classifiers (LR & MLP)
echo "--> Running ESM-2 embedding extraction and classifiers..."
$PYTHON_BIN -m deepise_ml.cli bench-esm2 --batch-size 32

# 2. Re-run unified evaluation across all 5 models
echo "--> Running unified evaluation across all 5 models..."
$PYTHON_BIN -m deepise_ml.cli eval-phase1

echo "=== ESM-2 Benchmark Finished at $(date) ==="

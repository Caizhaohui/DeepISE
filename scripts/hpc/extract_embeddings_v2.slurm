#!/usr/bin/env bash
#SBATCH --job-name=emb_v2
#SBATCH --partition=qgpu_4090
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/emb_v2_%j.log
#SBATCH --error=logs/emb_v2_%j.log

set -euo pipefail

echo "=== Running ESM-2 Embedding Extraction on $(hostname) at $(date) ==="
nvidia-smi

ENV_BIN="/hpcfs/fhome/caizhh/.conda/envs/DeepISE/bin"
export PATH="$ENV_BIN:$PATH"
export PYTHONPATH="python:${PYTHONPATH:-}"
export HF_ENDPOINT="https://hf-mirror.com"
PYTHON_BIN="$ENV_BIN/python"

# Extract ESM-2 35M embeddings
echo "--> Extracting ESM-2 35M embeddings..."
$PYTHON_BIN scripts/audit/extract_split_v2_embeddings.py --model facebook/esm2_t12_35M_UR50D --tag esm2_35m --batch-size 64

# Extract ESM-2 8M embeddings
echo "--> Extracting ESM-2 8M embeddings..."
$PYTHON_BIN scripts/audit/extract_split_v2_embeddings.py --model facebook/esm2_t6_8M_UR50D --tag esm2_8m --batch-size 128

echo "=== All Embeddings Extracted Successfully at $(date) ==="

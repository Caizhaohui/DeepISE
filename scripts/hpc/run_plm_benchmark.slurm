#!/bin/bash
#SBATCH --job-name=deepise_plm_bench
#SBATCH --partition=qgpu_4090
#SBATCH --nodelist=gnode19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3000M
#SBATCH --time=01:00:00
#SBATCH --output=logs/plm_benchmark_%j.log
#SBATCH --error=logs/plm_benchmark_%j.log

echo "=== DeepISE PLM Benchmark Starting on $(hostname) at $(date) ==="
source /hpcfs/fpublic/app/miniforge3/conda/etc/profile.d/conda.sh
conda activate DeepISE
export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONPATH="python:${PYTHONPATH:-}"

python scripts/benchmark_plm_models.py \
    --split-dir data/splits/cluster30 \
    --embeddings-dir data/processed/embeddings \
    --tables-dir benchmark/tables \
    --reports-dir benchmark/reports \
    --batch-size 32 \
    --device cuda

EXIT_CODE=$?
echo "=== DeepISE PLM Benchmark Finished with code $EXIT_CODE at $(date) ==="
exit $EXIT_CODE

#!/usr/bin/env bash
#SBATCH --job-name=eval_p1
#SBATCH --partition=qcpu_23i
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_p1_%j.log
#SBATCH --error=logs/eval_p1_%j.log

set -euo pipefail

echo "=== Running DeepISE Phase-1 Evaluation on $(hostname) at $(date) ==="
ENV_BIN="/hpcfs/fhome/caizhh/.conda/envs/DeepISE/bin"
export PATH="$ENV_BIN:$PATH"
export PYTHONPATH="python:${PYTHONPATH:-}"
PYTHON_BIN="$ENV_BIN/python"

$PYTHON_BIN -m deepise_ml.cli bench-hmmer --threads 4
$PYTHON_BIN -m deepise_ml.cli eval-phase1

echo "=== Evaluation Finished with code $? at $(date) ==="

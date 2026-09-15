#!/usr/bin/env bash
#SBATCH --job-name=p1_bench_v2
#SBATCH --partition=qcpu_18i
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/p1_bench_v2_%j.log
#SBATCH --error=logs/p1_bench_v2_%j.log

set -euo pipefail

echo "=== Running Phase-1 Benchmark v2 on $(hostname) at $(date) ==="
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-16}"

ENV_BIN="/hpcfs/fhome/caizhh/.conda/envs/DeepISE/bin"
export PATH="$ENV_BIN:$PATH"
export PYTHONPATH="python:${PYTHONPATH:-}"
PYTHON_BIN="$ENV_BIN/python"

$PYTHON_BIN scripts/audit/run_phase1_benchmark_v2.py

echo "=== Phase-1 Benchmark v2 Finished at $(date) ==="

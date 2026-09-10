#!/bin/bash
#SBATCH --job-name=pip_install_deepise
#SBATCH --partition=qcpu_23i
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pip_install_%j.log
#SBATCH --error=logs/pip_install_%j.log

set -e
echo "=== Installing Python dependencies on $(hostname) at $(date) ==="
ENV_PATH="/hpcfs/fhome/caizhh/.conda/envs/DeepISE"

"$ENV_PATH/bin/pip" install --trusted-host mirrors.aliyun.com \
    biopython \
    polars \
    pyarrow \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    pydantic \
    pyyaml \
    typer \
    rich \
    tqdm \
    pytest \
    pytest-cov \
    ruff \
    torch \
    transformers \
    -i https://mirrors.aliyun.com/pypi/simple/

echo "=== Verifying DeepISE Environment ==="
"$ENV_PATH/bin/python" -c "import torch, polars, pyarrow, sklearn, Bio, pydantic, typer; print('Environment Verification SUCCESSFUL!')"
"$ENV_PATH/bin/mmseqs" version
"$ENV_PATH/bin/blastp" -version | head -n 1
"$ENV_PATH/bin/hmmsearch" -h | head -n 2

echo "=== DeepISE Environment Ready at $(date) ==="

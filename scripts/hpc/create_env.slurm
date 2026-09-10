#!/bin/bash
#SBATCH --job-name=create_deepise_env
#SBATCH --partition=qcpu_23i
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/create_env_%j.log
#SBATCH --error=logs/create_env_%j.log

set -e
echo "=== Starting DeepISE Conda Environment Creation on $(hostname) at $(date) ==="
source /hpcfs/fpublic/app/miniforge3/conda/etc/profile.d/conda.sh

ENV_PATH="/hpcfs/fhome/caizhh/.conda/envs/DeepISE"

# Remove if partially created
rm -rf "$ENV_PATH"

echo "--> Step 1: Installing Python 3.11 and Bioinformatics tools (mmseqs2, blast, hmmer, mafft) into $ENV_PATH..."
/hpcfs/fpublic/app/miniforge3/conda/bin/mamba create -p "$ENV_PATH" -c conda-forge -c bioconda python=3.11 mmseqs2 blast hmmer mafft pip -y

echo "--> Step 2: Installing Python ML and data science packages via pip..."
"$ENV_PATH/bin/pip" install \
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
    -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "--> Step 3: Verifying environment..."
"$ENV_PATH/bin/python" -c "import torch, polars, pyarrow, sklearn, Bio, pydantic, typer; print('Environment Verification SUCCESSFUL!')"
"$ENV_PATH/bin/mmseqs" version

echo "=== DeepISE Environment Ready at $(date) ==="

#!/bin/bash
#SBATCH --job-name=install_phase0
#SBATCH --partition=qcpu_23i
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/install_phase0_%j.log
#SBATCH --error=logs/install_phase0_%j.log

set -e
echo "=== Installing Phase-0 Python packages into DeepISE on $(hostname) at $(date) ==="
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
    -i https://mirrors.aliyun.com/pypi/simple/

echo "=== Verifying DeepISE Environment ==="
"$ENV_PATH/bin/python" -c "import polars, pyarrow, sklearn, Bio, pydantic, typer; print('DeepISE Phase-0 Environment SUCCESSFUL!')"
"$ENV_PATH/bin/mmseqs" version

echo "=== DeepISE Environment Ready at $(date) ==="

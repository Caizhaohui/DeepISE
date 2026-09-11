#!/bin/bash
#SBATCH --job-name=install_phase1
#SBATCH --partition=qcpu_23i
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/install_phase1_%j.log
#SBATCH --error=logs/install_phase1_%j.log

set -e
echo "=== Installing Phase-1 ML/DL packages into DeepISE on $(hostname) at $(date) ==="
ENV_PATH="/hpcfs/fhome/caizhh/.conda/envs/DeepISE"

"$ENV_PATH/bin/pip" install --trusted-host mirrors.aliyun.com \
    transformers \
    accelerate \
    matplotlib \
    seaborn \
    -i https://mirrors.aliyun.com/pypi/simple/

echo "=== Verifying DeepISE Phase-1 Environment ==="
"$ENV_PATH/bin/python" -c "import torch, transformers, matplotlib, sklearn; print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available(), 'Transformers:', transformers.__version__)"

echo "=== DeepISE Phase-1 Environment Ready at $(date) ==="

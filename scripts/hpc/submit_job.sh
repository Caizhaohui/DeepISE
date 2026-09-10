#!/bin/bash
# Helper script to submit a task to SLURM queue
set -e

JOB_NAME="deepise_job"
PARTITION="qcpu_23i"
CPUS=4
MEM="8G"
TIME="02:00:00"
CMD=""
WAIT=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --name) JOB_NAME="$2"; shift ;;
        --partition) PARTITION="$2"; shift ;;
        --cpus) CPUS="$2"; shift ;;
        --mem) MEM="$2"; shift ;;
        --time) TIME="$2"; shift ;;
        --cmd) CMD="$2"; shift ;;
        --wait) WAIT=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$CMD" ]; then
    echo "Usage: $0 --cmd '<command>' [--name <name>] [--partition <part>] [--cpus <N>] [--mem <M>] [--wait]"
    exit 1
fi

mkdir -p logs

SCRIPT_FILE=$(mktemp logs/sbatch_tmp_XXXXXX.slurm)

cat << 'EOF' > "$SCRIPT_FILE"
#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH --partition=__PARTITION__
#SBATCH --cpus-per-task=__CPUS__
#SBATCH --mem=__MEM__
#SBATCH --time=__TIME__
#SBATCH --output=logs/__JOB_NAME___%j.log
#SBATCH --error=logs/__JOB_NAME___%j.log

echo "=== Job started on $(hostname) at $(date) ==="
source /hpcfs/fpublic/app/miniforge3/conda/etc/profile.d/conda.sh
conda activate DeepISE || conda activate /hpcfs/fhome/caizhh/.conda/envs/DeepISE
export PATH="/hpcfs/fhome/caizhh/.conda/envs/DeepISE/bin:$PATH"

__CMD__
EXIT_CODE=$?
echo "=== Job finished with code $EXIT_CODE at $(date) ==="
exit $EXIT_CODE
EOF

sed -i "s|__JOB_NAME__|$JOB_NAME|g" "$SCRIPT_FILE"
sed -i "s|__PARTITION__|$PARTITION|g" "$SCRIPT_FILE"
sed -i "s|__CPUS__|$CPUS|g" "$SCRIPT_FILE"
sed -i "s|__MEM__|$MEM|g" "$SCRIPT_FILE"
sed -i "s|__TIME__|$TIME|g" "$SCRIPT_FILE"
sed -i "s|__CMD__|$CMD|g" "$SCRIPT_FILE"

EXTRA_ARGS=""
if [ "$WAIT" = true ]; then
    EXTRA_ARGS="--wait"
fi

JOB_OUT=$(sbatch $EXTRA_ARGS "$SCRIPT_FILE")
JOB_ID=$(echo "$JOB_OUT" | awk '{print $NF}')
echo "Submitted SLURM job: $JOB_ID (Job: $JOB_NAME, Partition: $PARTITION)"
rm -f "$SCRIPT_FILE"

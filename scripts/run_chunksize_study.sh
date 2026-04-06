#!/bin/bash
#SBATCH -J rag_chunksize
#SBATCH -p NvidiaAll
#SBATCH -t 08:00:00

# Logs inside project
LOG_DIR="$HOME/rag_project/logs"
mkdir -p "$LOG_DIR"

LOG="$LOG_DIR/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-noid}.log"
exec > >(tee -a "$LOG") 2>&1

set -Eeuo pipefail
trap 'st=$?; echo "ERR on line $LINENO: $BASH_COMMAND (exit $st)"; exit $st' ERR

echo "==== START $(date) node=$(hostname) job=${SLURM_JOB_ID:-noid} ===="

# Activate conda
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER-}
conda activate AAML
set -u

python -V || true

# Temp dir (optional but good)
JOB_LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
JOB_TMP="$JOB_LOCAL_BASE/rag_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMP"
chmod 700 "$JOB_TMP"
export TMPDIR="$JOB_TMP"

echo "TMPDIR=$TMPDIR"

# Go to project
PROJECT_DIR="$HOME/rag_project"
SCRIPT_PATH="$PROJECT_DIR/notebooks/06_full_generation_evaluation_TopK_variation.py"

cd "$PROJECT_DIR"

# RUN EXPERIMENT
python "$SCRIPT_PATH"

echo "==== END $(date) ===="

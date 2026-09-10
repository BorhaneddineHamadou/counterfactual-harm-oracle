#!/bin/bash
#SBATCH --job-name=prox_p1
#SBATCH --output=logs/calibrate_%j_%a.out
#SBATCH --error=logs/calibrate_%j_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpuv100,gpup100,gpup6000
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
# node27 breaks the bridge (all calib-3 bridge_start_timeouts + a singularity
# hang that squats on the GPU after the worker self-kills)
#SBATCH --exclude=node27

# Proxima Phase-1 worker. Submit:
#   JOBS_FILE=... sbatch [--array=0-K] calibrate.slurm
# With an array, each task takes an equal slice of the job list.

ARRAY_IDX=${SLURM_ARRAY_TASK_ID:-0}
N_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
REPO_ROOT=${REPO_ROOT:-$HOME/counterfactual-harm-oracle}
OUTDIR=${STORE:-$REPO_ROOT/data/openpilot}
OPENPILOT_SIF=${OPENPILOT_SIF:-$HOME/openpilot_dev.sif}
OPENPILOT_ROOT=${OPENPILOT_ROOT:-$HOME/openpilot}
mkdir -p logs
mkdir -p "$OUTDIR"

export JOBS_FILE=${JOBS_FILE:?set JOBS_FILE}
N_JOBS=$(grep -c . "$JOBS_FILE")
PER=$(( (N_JOBS + N_TASKS - 1) / N_TASKS ))
export JOB_LO=$(( ARRAY_IDX * PER ))
export JOB_HI=$(( JOB_LO + PER ))
export RESULT_FILE="$OUTDIR/$(basename ${JOBS_FILE%.jsonl})_task${ARRAY_IDX}.results.jsonl"
export MANAGER_LOG="$OUTDIR/manager_${SLURM_JOB_ID}_${ARRAY_IDX}.log"
# threats must RENDER or openpilot cannot perceive them: calibrating without
# this tunes the BLIND SUT (rounds 1-5 did, which is why their ranges never
# trigger for the seeing ego)
export PROXIMA_RENDER_THREATS=1

PRIVATE_SHM="/tmp/shm_${SLURM_JOB_ID}_${ARRAY_IDX}"
mkdir -p "$PRIVATE_SHM"
DISPLAY_NUM=$((61 + ARRAY_IDX))

echo "=== calibration task ${ARRAY_IDX}: jobs ${JOB_LO}-${JOB_HI} of ${N_JOBS} ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

singularity exec --nv \
  --bind $HOME:$HOME \
  --bind $PRIVATE_SHM:/dev/shm \
  $OPENPILOT_SIF bash -c "
  export PATH=~/.local/bin:$OPENPILOT_ROOT/.venv/bin:\$PATH
  export DISPLAY=:${DISPLAY_NUM}
  export DISPLAY_NUM='$DISPLAY_NUM'
  export JOBS_FILE='$JOBS_FILE'
  export JOB_LO='$JOB_LO'
  export JOB_HI='$JOB_HI'
  export RESULT_FILE='$RESULT_FILE'
  export MANAGER_LOG='$MANAGER_LOG'
  export PROXIMA_RENDER_THREATS='$PROXIMA_RENDER_THREATS'
  export OPENPILOT_ROOT='$OPENPILOT_ROOT'
  cd $OPENPILOT_ROOT
  python3 $REPO_ROOT/adapters/metadrive_openpilot/worker.py
"

rm -rf "$PRIVATE_SHM"
echo "=== task ${ARRAY_IDX} finished: $(date) ==="

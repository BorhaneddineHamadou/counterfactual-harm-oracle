#!/bin/bash
#SBATCH --job-name=prox_camp
#SBATCH --output=logs/campaign_%j_%a.out
#SBATCH --error=logs/campaign_%j_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpuv100,gpup100,gpup6000
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
# node27 breaks the bridge (all calib-3 bridge_start_timeouts + a singularity
# hang that squats on the GPU after the worker self-kills)
#SBATCH --exclude=node27

# Proxima full-campaign worker (long walltime, 5 engagement retries).
# Submit: JOBS_FILE=... sbatch --array=0-11 campaign.slurm

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
export MANAGER_LOG="$OUTDIR/manager_camp_${SLURM_JOB_ID}_${ARRAY_IDX}.log"
export PROXIMA_RETRIES=5

PRIVATE_SHM="/tmp/shm_${SLURM_JOB_ID}_${ARRAY_IDX}"
mkdir -p "$PRIVATE_SHM"
# DISPLAY_BASE overridable so a second array can run on the same nodes
# without Xvfb display collisions (sbatch snapshots this script, so
# editing it does not affect already-submitted jobs)
DISPLAY_NUM=$(( ${DISPLAY_BASE:-41} + ARRAY_IDX ))

echo "=== campaign task ${ARRAY_IDX}: jobs ${JOB_LO}-${JOB_HI} of ${N_JOBS} ==="
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
  export PROXIMA_RETRIES='$PROXIMA_RETRIES'
  export OPENPILOT_ROOT='$OPENPILOT_ROOT'
  cd $OPENPILOT_ROOT
  python3 $REPO_ROOT/adapters/metadrive_openpilot/worker.py
"

rm -rf "$PRIVATE_SHM"
echo "=== task ${ARRAY_IDX} finished: $(date) ==="

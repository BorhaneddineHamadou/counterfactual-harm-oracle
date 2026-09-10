#!/bin/bash
#SBATCH --job-name=tf_camp
#SBATCH --partition=gpuv100,gpup6000,classicgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=node27
#SBATCH --output=%x_%A_%a.out

# Calibration worker: array task N runs its slice of $JOBS_FILE against its
# own CARLA server (port spaced by 50 so tasks can share a node).
# Submit: JOBS_FILE=... sbatch --array=0-3 calib.slurm
set -u
: "${JOBS_FILE:?set JOBS_FILE=<path to jobs jsonl>}"
TASK=${SLURM_ARRAY_TASK_ID:-0}
# NTASK_OVERRIDE lets a partial resubmission (e.g. --array=23-28) keep the
# original slice stride instead of SLURM_ARRAY_TASK_COUNT of the new array.
NTASK=${NTASK_OVERRIDE:-${SLURM_ARRAY_TASK_COUNT:-1}}
PORT=$((2000 + TASK * 50))

REPO_ROOT=${REPO_ROOT:-$HOME/counterfactual-harm-oracle}
ADAPTER=$REPO_ROOT/adapters/carla_transfuser
CARLA_ROOT=~/carla_09101
VENV=~/tfuse_venv
SIF=~/sifs/python37full.sif
EGG=$(ls $CARLA_ROOT/PythonAPI/carla/dist/carla-*-py3.7-linux-x86_64.egg)

# CARLA server with the libnvidia-gpucomp --nv workaround + restart loop
GPUCOMP=$(ls /usr/lib64/libnvidia-gpucomp.so.* | head -1)
echo "=== task $TASK/$NTASK port $PORT (gpucomp: $GPUCOMP) ==="
(while true; do
  singularity exec --nv \
    --bind "$GPUCOMP":/nvidia_extra/$(basename "$GPUCOMP") \
    ~/sifs/carla09101.sif /bin/bash -c \
    "export LD_LIBRARY_PATH=/nvidia_extra:/.singularity.d/libs:\$LD_LIBRARY_PATH
     DISPLAY= SDL_VIDEODRIVER=offscreen /home/carla/CarlaUE4.sh -opengl \
     -nosound -carla-rpc-port=$PORT -quality-level=Epic"
  echo "[server-loop] CARLA exited $? -- restarting in 5s"
  sleep 5
done) > tf_calib_server_${SLURM_ARRAY_JOB_ID:-0}_${TASK}.log 2>&1 &
sleep 45

singularity exec --nv \
  --bind /scratch,$CARLA_ROOT,$HOME \
  $SIF env \
  PYTHONPATH=$EGG:$CARLA_ROOT/PythonAPI/carla:~/transfuser_repo/leaderboard:~/transfuser_repo/scenario_runner:~/transfuser_repo/team_code_transfuser:$ADAPTER \
  LD_LIBRARY_PATH=$HOME/carla_libs:/.singularity.d/libs \
  CARLA_ROOT=$CARLA_ROOT \
  TF_CKPT=~/transfuser_assets/model_ckpt/transfuser \
  CARLA_PORT=$PORT \
  CARLA_EXTERNAL=1 \
  $VENV/bin/python $ADAPTER/worker.py $JOBS_FILE $TASK $NTASK
pkill -u $USER -f "carla-rpc-port=$PORT" 2>/dev/null || true
echo "=== task $TASK done ==="

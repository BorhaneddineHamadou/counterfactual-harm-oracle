#!/bin/bash
# One-time environment setup for the TransFuser adapter (login node, has egress).
set -e
ASSETS=~/transfuser_assets
CARLA_DIR=~/carla_09101
REPO=~/transfuser_repo
VENV=~/tfuse_venv
SIF=~/sifs/python37.sif

echo "== 1. CARLA binaries =="
if [ ! -d "$CARLA_DIR" ]; then
  mkdir -p "$CARLA_DIR" && cd "$CARLA_DIR"
  tar -xf "$ASSETS/CARLA_0.9.10.1.tar.gz"
  tar -xf "$ASSETS/AdditionalMaps_0.9.10.1.tar.gz"
fi

echo "== 2. model checkpoints =="
if [ ! -d "$ASSETS/model_ckpt" ]; then
  cd "$ASSETS" && unzip -q models_2022.zip
  # repo README: extract into model_ckpt/ (zip layout may already contain it)
  [ -d model_ckpt ] || { mkdir model_ckpt && mv models_2022/* model_ckpt/; }
fi

echo "== 3. python 3.7 SIF =="
mkdir -p ~/sifs
[ -f "$SIF" ] || singularity pull "$SIF" docker://python:3.7-slim

echo "== 4. venv (built with the SIF's python; used inside the SIF) =="
if [ ! -d "$VENV" ]; then
  singularity exec "$SIF" python3.7 -m venv "$VENV"
  singularity exec "$SIF" "$VENV/bin/pip" install --upgrade pip
  singularity exec "$SIF" "$VENV/bin/pip" install \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    torch==1.12.1+cu113 torchvision==0.13.1+cu113
  # agent-runtime subset (full requirements.txt is training-heavy)
  singularity exec "$SIF" "$VENV/bin/pip" install \
    numpy==1.21.6 opencv-python-headless==4.6.0.66 Pillow==9.2.0 \
    timm==0.6.7 Shapely==1.8.4 filterpy==1.4.5 py-trees==0.8.3 \
    networkx==2.6.3 dictor==0.1.10 ephem==4.1.3 tabulate simple-watchdog-timer \
    scipy==1.7.3 pygame==2.1.2 transforms3d
fi

echo "== 5. CARLA python egg (py3.7) =="
EGG=$(ls "$CARLA_DIR"/PythonAPI/carla/dist/carla-*-py3.7-linux-x86_64.egg)
echo "egg: $EGG  (exported via PYTHONPATH in slurm/worker)"

echo "== done. next: sbatch campaign.slurm (see README) =="

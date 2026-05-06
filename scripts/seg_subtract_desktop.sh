#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
IMGPATH=$1
SAVEDIR=$2
SAVE_ANGLE=1
SAVE_SUB=1
segdir="${3:-""}"
par_fpath="${4:-"parameters.yml"}"
par_fname="$(basename -- "$par_fpath")"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
for d in "$CONDA_PREFIX"/lib/python3.12/site-packages/nvidia/*/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
done


export SAVEDIR=$SAVEDIR
export SAVE_ANGLE=$SAVE_ANGLE
export SAVE_SUB=$SAVE_SUB
export SEGMENTATION_DIR=$segdir
export PAR_FPATH="${SAVEDIR}/parameters.yml"
bash scripts/seg_subtract_v1.sh "${IMGPATH}"
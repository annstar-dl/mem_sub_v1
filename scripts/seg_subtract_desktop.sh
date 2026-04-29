#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
IMGPATH=$1
SAVEDIR=$2
SAVE_ANGLE=1
SAVE_SUB=1
segdir="${3:-""}"

if [[ -f "${SAVEDIR}" ]]; then
  echo "The directory ${SAVEDIR} already exists. Please choose a different directory or remove the existing one."
  exit 1
fi
mkdir -p "$SAVEDIR"
cp "parameters.yml" "${SAVEDIR}/parameters.yml"

if [ -n "${segdir}" ]; then
    SEGMENTATION_DIR="${segdir%/}"
    export SEGMENTATION_DIR
    echo  "Using segmentation model from: ${SEGMENTATION_DIR}"
fi


export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
for d in "$CONDA_PREFIX"/lib/python3.12/site-packages/nvidia/*/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
done


export SAVEDIR=$SAVEDIR
export SAVE_ANGLE=$SAVE_ANGLE
export SAVE_SUB=$SAVE_SUB
bash scripts/seg_subtract_v1.sh "${IMGPATH}"
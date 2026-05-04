#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
IMGPATH=$1
SAVEDIR=$2
SAVE_ANGLE=1
SAVE_SUB=1
segdir="${3:-""}"


mkdir -p "$SAVEDIR"
if [[ -f "${SAVEDIR}/parameters.yml" ]]; then
  #chekc if "${SAVE_DIR_PATH}/parameters.yml" is the same as "parameters.yml" in the current directory, if not copy the new one
  #if not copy the exit the code and request to fix the parameters.yml file
  #in the future change this to use old "${SAVE_DIR_PATH}/parameters.yml" as parameters.yml file, when the path to that file becomes an argument
  if ! cmp -s "parameters.yml" "${SAVEDIR}/parameters.yml"; then
    echo "Error: ${SAVEDIR}/parameters.yml already exists and is different from the current parameters.yml. Please fix this before running the script."
    exit 1
  else
    echo "parameters.yml already exists in ${SAVEDIR} and is the same as the current parameters.yml. No need to copy."
  fi
else
  cp "parameters.yml" "${SAVEDIR}/parameters.yml"
fi


if [[ ! -f "${SAVEDIR}/exp_config.yml" ]]; then
  python "scripts/record_hash.py" -sp "${SAVEDIR}"
fi

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
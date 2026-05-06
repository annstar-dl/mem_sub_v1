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

mkdir -p "$SAVEDIR"
if [[ -f "${SAVEDIR}/parameters.yml" ]]; then
  #chekc if ${par_fpath} is the same as parameters.yml in the current directory
  if ! cmp -s ${par_fpath} "${SAVEDIR}/parameters.yml"; then
    echo "Error: ${par_fname} already exists and is different from the current parameters.yml. Please fix this before running the script."
    exit 1
  else
    echo "parameters file already exists in ${SAVEDIR} and is the same as the current parameters.yml. No need to copy."
  fi
  else
  cp ${par_fpath} "${SAVEDIR}/parameters.yml"
fi

#record the current commit hash in a yml file in the savedir if it doesn't already exist, this is useful for later reference and to avoid confusion
python -m mem_sub.record_hash -sp "${SAVEDIR}"




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
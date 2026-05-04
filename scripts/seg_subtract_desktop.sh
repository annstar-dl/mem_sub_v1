#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
IMGPATH=$1
SAVEDIR=$2
SAVE_ANGLE=1
SAVE_SUB=1
segdir="${3:-""}"
par_fpath="${4:-"parameters.yml"}"


mkdir -p "$SAVEDIR"
if [[ -f "${SAVEDIR}/${par_fpath}" ]]; then
  #chekc if "${SAVE_DIR_PATH}/${par_fpath}" is the same as ${par_fpath} in the current directory, if not copy the new one
  #if not copy the exit the code and request to fix the ${par_fpath} file
  #in the future change this to use old "${SAVE_DIR_PATH}/${par_fpath}" as ${par_fpath} file, when the path to that file becomes an argument
  if ! cmp -s ${par_fpath} "${SAVEDIR}/${par_fpath}"; then
    echo "Error: ${SAVEDIR}/${par_fpath} already exists and is different from the current ${par_fpath}. Please fix this before running the script."
    exit 1
  else
    echo "${par_fpath} already exists in ${SAVEDIR} and is the same as the current ${par_fpath}. No need to copy."
  fi
  else
  cp ${par_fpath} "${SAVEDIR}/${par_fpath}"
fi

#record the current commit hash in a yml file in the savedir if it doesn't already exist, this is useful for later reference and to avoid confusion
python "scripts/record_hash.py" -sp "${SAVEDIR}"




export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
for d in "$CONDA_PREFIX"/lib/python3.12/site-packages/nvidia/*/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
done


export SAVEDIR=$SAVEDIR
export SAVE_ANGLE=$SAVE_ANGLE
export SAVE_SUB=$SAVE_SUB
export SEGMENTATION_DIR=$segdir
export PAR_FPATH=$par_fpath
bash scripts/seg_subtract_v1.sh "${IMGPATH}"
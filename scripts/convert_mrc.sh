#!/bin/bash

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 MRC_PATH SAVE_DIR SAVE_HASH_FLAG"
    exit 1
fi

MRCPATH=$1
SAVEDIR=$2
SAVE_HASH_FLAG=$3

if [[ $SAVE_HASH_FLAG -eq 1 ]]; then
    python scripts/record_hash.py -dp "${SAVEDIR}"
fi

if [[ -d "$MRCPATH" ]]; then
  MRCDIR=$(basename "$MRCPATH")
  DS_MICROGRAPHS_PATH="${SAVEDIR}/${MRCDIR}_jpg_ds"
  SAVEDIR_MRC=$DS_MICROGRAPHS_PATH
elif [[ -f "$MRCPATH" ]]; then
  if [[ "${MRCPATH,,}" == *.mrc ]]; then
    #cut the mrc extension from the filename
    FILENAME="$(basename -- "$MRCPATH")"; FILENAME="${FILENAME::-4}"
    echo "Processing file: ${MRCPATH}"
  else
    echo "Error: ${MRCPATH} is not an mrc file."
    exit 1
  fi
  SAVEDIR_MRC="${SAVEDIR}/${MRCDIR}_jpg_ds"
fi

echo "SAVEDIR_MRC is ${SAVEDIR_MRC}"
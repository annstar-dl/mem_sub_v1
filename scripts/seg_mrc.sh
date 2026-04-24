#!/bin/bash

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 MRC_PATH SAVE_DIR SAVE_HASH_FLAG"
    exit 1
fi

MRCPATH=$1
SAVEDIR=$2
SAVE_HASH_FLAG=$3
segmentation_dir=$(python -c "import yaml; cfg=yaml.safe_load(open('seg_parameters.yml')); print(cfg['model_dir'])")
SEGMENTATION_DIR=${segmentation_dir%/}
echo  "Using segmentation model from: ${SEGMENTATION_DIR}"

if [[ $SAVE_HASH_FLAG -eq 1 ]]; then
    python scripts/record_hash.py -dp "${SAVEDIR}" --save_seg_dir
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

  INPUTDIR=$(dirname "$MRCPATH")
  MRCDIR=$(basename "$INPUTDIR")
  SAVEDIR_MRC="${SAVEDIR}/${MRCDIR}_jpg_ds"
  DS_MICROGRAPHS_PATH="${SAVEDIR_MRC}/${FILENAME}.jpg"
fi

echo "SAVEDIR_MRC is ${SAVEDIR_MRC}"
# Convert mrc files to jpg for segmentation
python "tools/mrc2image.py" "${MRCPATH}" \
                            -o "${SAVEDIR_MRC}" --format "jpg" -dsa --scale --sub_mean --border_size 7


python "membrane_seg/seg_onnx.py" \
--model_dir "${SEGMENTATION_DIR}" \
--onnx_fname model.onnx \
--data_path ${DS_MICROGRAPHS_PATH} \
--save_dir "${SAVEDIR}"
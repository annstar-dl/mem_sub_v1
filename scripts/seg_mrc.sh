#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 MRC_PATH SAVE_DIR [segmentation_dir]"
    exit 1
fi

MRCPATH=$1
SAVEDIR=$2
segmentation_dir="${3:-"membrane_seg/seg_model/mem_mad_2026_march_warmup_lr_0005_500000"}"
SEGMENTATION_DIR=${segmentation_dir%/}
echo  "Using segmentation model from: ${SEGMENTATION_DIR}"

# if child folder is not misc, create the folder and copy the parameters.yml file
# else this script is being called by seg_subtract_v1.sh and the parameters.yml file is already in the misc folder, so we don't need to copy it again
child_folder="$(basename -- "$SAVEDIR")"
if [[ "$child_folder" != "misc" ]]; then
    if [[ -d "${SAVEDIR}" ]]; then
      echo "The directory ${SAVEDIR} already exists. Please choose a different directory or remove the existing one."
      exit 1
    fi
    mkdir -p "$SAVEDIR"
    cp "parameters.yml" "${SAVEDIR}/parameters.yml"
fi

#save the segmentation model path in a text file in the savedir if it doesn't already exist, this is useful for later reference and to avoid confusion about which model was used for segmentation
if [[ ! -f "${SAVEDIR}/seg_model.txt" ]]; then
  echo "segmentation model: ${SEGMENTATION_DIR}" > "${SAVEDIR}/seg_model.txt"
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
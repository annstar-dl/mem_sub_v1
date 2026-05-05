#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 MRC_PATH SAVE_DIR [segmentation_dir]"
    exit 1
fi

MRCPATH=$1
SAVEDIR=$2
SEGMENTATION_DIR=$3
par_fpath="${4:-"parameters.yml"}"

if [[ -z "${SEGMENTATION_DIR}" ]]; then
  SEGMENTATION_DIR="membrane_seg/seg_model/mem_mad_2026_march_warmup_lr_0005_500000"
fi

SEGMENTATION_DIR=${SEGMENTATION_DIR%/}
echo  "Using segmentation model from: ${SEGMENTATION_DIR}"
mkdir -p "$SAVEDIR"
# if child folder is not misc, create the folder and copy the parameters.yml file
# else this script is being called by seg_subtract_v1.sh and the parameters.yml file is already in the misc folder, so we don't need to copy it again
child_folder="$(basename -- "$SAVEDIR")"
if [[ "$child_folder" != "misc" ]]; then

    if [[ -f "${SAVEDIR}/${par_fpath}" ]]; then
      #chekc if "${SAVE_DIR_PATH}/parameters.yml" is the same as "parameters.yml" in the current directory, if not copy the new one
      #if not copy the exit the code and request to fix the parameters.yml file
      #in the future change this to use old "${SAVE_DIR_PATH}/parameters.yml" as parameters.yml file, when the path to that file becomes an argument
      if ! cmp -s ${par_fpath} "${SAVEDIR}/${par_fpath}"; then
        echo "Error: ${SAVEDIR}/${par_fpath} already exists and is different from the current ${par_fpath}. Please fix this before running the script."
      exit 1
      else
        echo "${par_fpath} already exists in ${SAVEDIR} and is the same as the current ${par_fpath}. No need to copy."
      fi
    else
      cp ${par_fpath} "${SAVEDIR}/${par_fpath}"
    fi
    #save commit hash of the current code in a yml file in the savedir if it doesn't already exist, this is useful for later reference and to avoid confusion
    python -m mem_sub.record_hash -sp "${SAVEDIR}"
fi

#save the segmentation model path in a text file in the savedir if it doesn't already exist, this is useful for later reference and to avoid confusion about which model was used for segmentation
if [[ ! -f "${SAVEDIR}/seg_model.txt" ]]; then
  echo "segmentation model: ${SEGMENTATION_DIR}" > "${SAVEDIR}/seg_model.txt"
else
    expected_content="segmentation model: ${SEGMENTATION_DIR}"
    existing_content=$(cat "${SAVEDIR}/seg_model.txt")
  if [[ "$expected_content" != "$existing_content" ]]; then
    echo "Error: ${SAVEDIR}/seg_model.txt already exists and is different from the current segmentation model. Please fix this before running the script."
    exit 1
  else
    echo "seg_model.txt already exists in ${SAVEDIR} and is the same as the current segmentation model. No need to copy."
  fi
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
                            -o "${SAVEDIR_MRC}" \
                            --format "jpg" \
                            -dsa --scale --sub_mean \
                            --border_size 7 --par_fpath ${par_fpath}


python "membrane_seg/seg_onnx.py" \
--model_dir "${SEGMENTATION_DIR}" \
--onnx_fname model.onnx \
--data_path ${DS_MICROGRAPHS_PATH} \
--save_dir "${SAVEDIR}"
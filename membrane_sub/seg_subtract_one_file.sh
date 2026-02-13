#!/bin/bash

FILENAME=$1
echo "Processing file: ${FILENAME}"
echo "nvidia-smi output:"
nvidia-smi
# Normalize INPUTDIR and SAVEDIR: remove trailing slash if present
# This ensures basename and later path concatenations work whether the user
# provides "/path/to/dir" or "/path/to/dir/".
if [ -n "${INPUTDIR}" ]; then
    INPUTDIR="${INPUTDIR%/}"
fi
if [ -n "${SAVEDIR}" ]; then
    SAVEDIR="${SAVEDIR%/}"
fi

# extract name of the dataset from the path
MRC_DIR=$(basename "$INPUTDIR")
# create a new directory to store the results
mkdir -p ${SAVEDIR}
mkdir -p "${SAVEDIR}/misc/${MRC_DIR}"
# go into the new directory
# copy input files directory to an output directory
cp "${INPUTDIR}/${FILENAME}.mrc" "${SAVEDIR}/misc/${MRC_DIR}/${FILENAME}.mrc"
# Convert mrc files to jpg for segmentation
python "mrc2image.py" "${SAVEDIR}/misc/${MRC_DIR}" \
                            -o "${SAVEDIR}/misc" --format "jpg" -dsa --scale -fn "${FILENAME}.mrc"
DS_MICROGRAPHS_PATH="${SAVEDIR}/misc/${MRC_DIR}_jpg_ds/${FILENAME}.jpg"
python "seg_onnx.py" \
--model_dir "${SEGMENTATION_DIR}" \
--onnx_fname model.onnx \
--data_path ${DS_MICROGRAPHS_PATH} \
--save_dir "${SAVEDIR}/misc"

# Subtract the predicted masks from the original micrographs
if [ $SAVE_ANGLE -eq 1 ] && [ $SAVE_SUB -eq 1 ]; then
    python "run_mrc_subtraction.py" \
 -dp ${SAVEDIR} -ip "${SAVEDIR}/misc/${MRC_DIR}" \
 --out_format_sub "mrc" "png" \
  --out_format_mem "mrc" "png" \
  --save_angle \
  -do_sub \
  -fn "${FILENAME}.mrc"
else
    if [ $SAVE_ANGLE -eq 1 ]; then
            python "run_mrc_subtraction.py" \
   -dp ${SAVEDIR} -ip "${SAVEDIR}/misc/${MRC_DIR}" \
   --out_format_sub "mrc" "png" \
    --out_format_mem "mrc" "png" \
    --save_angle \
    -fn "${FILENAME}.mrc"
    elif [ $SAVE_SUB -eq 1 ]; then
        python "run_mrc_subtraction.py" \
   -dp ${SAVEDIR} -ip "${SAVEDIR}/misc/${MRC_DIR}" \
   --out_format_sub "mrc" "png" \
    --out_format_mem "mrc" "png" \
    -do_sub \
    -fn "${FILENAME}.mrc"
    else
        python "run_mrc_subtraction.py" \
   -dp ${SAVEDIR} -ip "${SAVEDIR}/misc/${MRC_DIR}" \
   --out_format_sub "mrc" "png" \
    --out_format_mem "mrc" "png" \
    -fn "${FILENAME}.mrc"
    fi
fi

# delete copied mrc file to save space
rm "${SAVEDIR}/misc/${MRC_DIR}/${FILENAME}.mrc"

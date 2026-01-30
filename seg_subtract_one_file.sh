#!/bin/bash

FILENAME=$1
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
                            -o "${SAVEDIR}" --format "jpg" -dsa --scale -fn "${FILENAME}.mrc"
DS_MICROGRAPHS_PATH="${SAVEDIR}/misc/${MRC_DIR}_jpg_ds"
python "seg_onnx.py" \
--model_dir "${SEGMENTATION_DIR}" \
--onnx_fname model.onnx \
--data_path ${DS_MICROGRAPHS_PATH} \
--save_dir "${SAVEDIR}/misc" \
-fn "${FILENAME}.jpg"

# Subtract the predicted masks from the original micrographs
python "run_mrc_subtraction.py" \
 -dp ${SAVEDIR} -ip "${SAVEDIR}/misc/${MRC_DIR}" \
 --out_format "mrc" "png" \
  --save_angle \
 -fn "${FILENAME}.mrc"
# delete copied mrc file to save space
rm "${SAVEDIR}/${MRC_DIR}/${FILENAME}.mrc"
echo "Subtracted micrographs are saved in ${SAVEDIR}/subtracted_mrc"



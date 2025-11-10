#!/bin/bash

FILENAME=$1
#extract name of the dataset from the path
MRC_DIR=$(basename "$INPUTDIR")
# create a new directory to store the results
mkdir -p $SAVEDIR
#go into the new directory
cd $SAVEDIR
# copy input files directory to an output directory
cp "${INPUTDIR}/${FILENAME}.mrc" "${SAVEDIR}/${MRC_DIR}/${FILENAME}.mrc"
# Convert mrc files to jpg for segmentation
python "mrc2image.py" "${SAVEDIR}/${MRC_DIR}" \
                            -o "${SAVEDIR}" --format "jpg" -dsa --scale -fn "${FILENAME}.mrc"
DS_MICROGRAPHS_PATH="${SAVEDIR}/images_jpg/${MRC_DIR}_ds"
python "seg_onnx.py" \
--model_dir "${SEGMENTATION_DIR}" \
--onnx_fname model.onnx \
--data_path ${DS_MICROGRAPHS_PATH} \
--save_dir "${SAVEDIR}" \
-fn "${FILENAME}.jpg"

# Subtract the predicted masks from the original micrographs
python "run_mrc_subtraction.py" \
 -dp ${SAVEDIR} -ip "${SAVEDIR}/${MRC_DIR}" \
 --out_format "mrc" "png" \
 -fn "${FILENAME}.mrc"
echo "Subtracted masks from original micrographs and saved results in ${SAVEDIR}/reconstructions/subtracted_png/${FILENAME}"





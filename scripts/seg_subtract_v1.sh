#!/bin/bash

IMGPATH=$1
# create a new directory to store the results
# Normalize SAVEDIR: remove trailing slash if present
# This ensures basename and later path concatenations work whether the user
# provides "/path/to/dir" or "/path/to/dir/".
if [ -n "${SAVEDIR}" ]; then
    SAVEDIR="${SAVEDIR%/}"
fi
mkdir -p ${SAVEDIR}

if [[ -d "$IMGPATH" ]]; then
    echo "Processing files in directory: ${IMGPATH}"
    INPUTDIR="$IMGPATH"
    if [ -n "${INPUTDIR}" ]; then
      INPUTDIR="${INPUTDIR%/}"
    fi
    # extract name of the dataset from the path
    MRC_DIR=$(basename "$INPUTDIR")
    PROCESS_DIR=1
    COPIED_MRCPATH="${INPUTDIR}/misc/${MRC_DIR}"
elif [[ -f "$IMGPATH" ]]; then
    echo "Processing file: ${IMGPATH}"
    INPUTDIR=$(dirname "$IMGPATH")
    # extract name of the dataset from the path
    MRC_DIR=$(basename "$INPUTDIR")
    FILENAME="$(basename -- "$IMGPATH")"; FILENAME="${FILENAME::-4}"
    PROCESS_DIR=0
    COPIED_MRCPATH="${INPUTDIR}/misc/${MRC_DIR}/${FILENAME}.mrc"
else
    echo "${IMGPATH} does not exist or is not a regular file/directory"
fi


echo "nvidia-smi output:"
nvidia-smi

bash scripts/seg_mrc.sh "${IMGPATH}" "${SAVEDIR}/misc"
echo "Segmentation completed. Saved results in ${SAVEDIR}/misc"
# Subtract the predicted masks from the original micrographs
if [[ $SAVE_ANGLE -eq 1 ]] && [[ $SAVE_SUB -eq 1 ]]; then
    python "tools/run_mrc_subtraction.py" \
 -dp ${SAVEDIR} -ip "${IMGPATH}" \
 --out_format_sub "mrc" "png" \
  --out_format_mem "mrc" "png" "npy"\
  --save_angle \
  --save_subtraction
else
    if [[ $SAVE_ANGLE -eq 1 ]]; then
            python "tools/run_mrc_subtraction.py" \
   -dp ${SAVEDIR} -ip "${IMGPATH}" \
   --out_format_sub "mrc" "png" \
    --out_format_mem "mrc" "png" \
    --save_angle

    elif [[ $SAVE_SUB -eq 1 ]]; then
        python "tools/run_mrc_subtraction.py" \
   -dp ${SAVEDIR} -ip "${IMGPATH}" \
   --out_format_sub "mrc" "png" \
    --out_format_mem "mrc" "png" \
    --save_subtraction

    else
        python "tools/run_mrc_subtraction.py" \
   -dp ${SAVEDIR} -ip "${IMGPATH}" \
   --out_format_sub "mrc" "png" \
    --out_format_mem "mrc" "png"
    fi
fi

if [[ $PROCESS_DIR -eq 1 ]]; then
    echo "Segmentation and subtraction completed for directory: ${INPUTDIR}. Saved results in ${SAVEDIR}/misc/${MRC_DIR}"
else
    echo "Segmentation and subtraction completed for file: ${IMGPATH}. Saved results in ${SAVEDIR}/misc/${MRC_DIR}/${FILENAME}"
fi

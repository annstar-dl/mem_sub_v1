#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# Path to the membrane detection project directory
MEMBRANE_DETECTION_DIR="/home/astar/Projects/membrane_detection"
# Path to the pretrained segmentation model directory
SEGMENTATION_DIR="/home/astar/Projects/membrane_detection/out/unet_membrane_256x256_500000"
MEMBRANE_SUBTRACTION_DIR="/home/astar/Projects/VesicleProjection" # Path to the VesicleProjection project directory
INPUT_FILE_FORMAT="jpg"  # Change to "tif" if input files are in TIFF format
SEGMENTATION_MODEL_FORMAT="paddleseg" # "onnx" or "paddleseg"
PADDLESEG_DIR="/home/astar/Projects/PaddleSeg" # Path to the PaddleSeg project directory
# Check if the directory argument is provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <output_directory_path> <input_files_directory>"
    exit 1
fi

MRC_DIR=$(basename "$2")
cd $1
# Subtract the predicted masks from the original micrographs
conda run -n ves_seg python "${MEMBRANE_SUBTRACTION_DIR}/run_mrc.py" \
 -dp ${PWD} -id ${MRC_DIR} -md "labels_dilated" \
 --in_format ${INPUT_FILE_FORMAT} \
 --out_format "mrc" "png"
echo "Subtracted masks from original micrographs and saved results in $PWD/reconstructions"
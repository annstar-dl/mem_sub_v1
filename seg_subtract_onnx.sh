#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2

MEMBRANE_DETECTION_DIR="/home/astar/Projects/membrane_detection"
SEGMENTATION_DIR="${MEMBRANE_DETECTION_DIR}/out/unet_membrane_256x256_noise_20_notonlybg_1000000"
MEMBRANE_SUBTRACTION_DIR="/home/astar/Projects/VesicleProjection"
# Check if the directory argument is provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <output_directory_path> <input_files_directory> <downsample_factor> <input_file_format>"
    exit 1
fi

MRC_DIR=$(basename "$2")
# create a new directory to store the results
mkdir -p $1
#go into the new directory
cd $1
# copy input files directory to an output directory
cp -r $2 .
echo "Copied MRC files from $2 to $PWD/$MRC_DIR"
# create a jpg directory to store the converted images
mkdir -p "images_jpg"
# create a directory to store predicted masks
mkdir -p "labels"

# Convert mrc files to jpg for segmentation

echo "Converting MRC files to JPG..."
echo "Using Membrane Subtraction directory: $PWD/${MRC_DIR}"
conda run -n ves_seg python "${MEMBRANE_SUBTRACTION_DIR}/mrc2jpg.py" "$PWD/${MRC_DIR}" -o "$PWD/images_jpg" --format "jpeg" -ds $3
echo "Converted MRC files to JPG in $PWD/images_jpg"

# Run the segmentation model
# infer
conda run -n paddleseg_onnx python "$MEMBRANE_DETECTION_DIR/export_onnx/run_onnx.py" \
--model_dir "${SEGMENTATION_DIR}/result" \
--onnx_fname model.onnx \
--data_path "${PWD}/images_jpg" \
--save_dir "${PWD}"
echo "Segmented images saved to $PWD/labels_pseudcolor"
# Subtract the predicted masks from the original micrographs

# run the membrane subtraction script
conda run -n ves_seg python "${MEMBRANE_SUBTRACTION_DIR}/run.py"  -dp ${PWD} -id ${MRC_DIR} -md "labels" --in_format $4 --out_format "mrc" -ds $3
echo "Subtracted masks from original micrographs and saved results in $PWD/reconstructions"


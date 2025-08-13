#!/usr/bin/env bash

# Initialize Conda in the script's shell environment
eval "$(conda shell.bash hook)"
# create a new directory to store the results
# Check if the directory argument is provided
PADDLESEG_DIR="/home/astar/Projects/PaddleSeg"
SEGMENTATION_DIR="/home/astar/Projects/membrane_detection/out/unet_membrane_256x256_noise_20_notonlybg_1000000"
MEMBRANE_SUBTRACTION_DIR="/home/astar/Projects/VesicleProjection"

if [ $# -lt 3 ]; then
    echo "Usage: $0 <output_directory_path> <mrc_files_directory> <downsample_factor>"
    echo "Example: $0 /path/to/output /path/to/mrc_files 4"
    exit 1
fi

MRC_DIR=$(basename "$2")

mkdir -p $1
#go into the new directory
cd $1
# copy mrc files directory to a new directory
#cp -r $2 .
echo "Copied MRC files from $2 to $PWD/$MRC_DIR"
# create a jpg directory to store the converted images
mkdir -p "images_jpg"
# create a directory to store predicted masks
mkdir -p "labels"

# Convert mrc files to jpg for segmentation
conda activate ves_seg
echo "Converting MRC files to JPG..."
echo "Using Membrane Subtraction directory: $PWD/${MRC_DIR}"
python "${MEMBRANE_SUBTRACTION_DIR}/mrc2jpg.py" "$PWD/${MRC_DIR}" -o "$PWD/images_jpg" --format "jpeg" -ds $3
echo "Converted MRC files to JPG in $PWD/images_jpg"
conda deactivate
# Run the segmentation model
conda activate paddleseg
# infer
python "${PADDLESEG_DIR}/deploy/python/infer.py" \
--config "${SEGMENTATION_DIR}/result/deploy.yaml" \
--image_path "$PWD/images_jpg" \
--save_dir "$PWD/label_pseudcolor"
echo "Segmented images saved to $PWD/label_pseudcolor"
conda deactivate
# Subtract the predicted masks from the original micrographs
conda activate ves_seg
# convert the pseudo-color masks to binary masks
python "${MEMBRANE_SUBTRACTION_DIR}/color2binary.py" "$PWD/label_pseudcolor"
echo "Converted pseudo-color masks to binary masks in $PWD/labels"
# run the membrane subtraction script
python "${MEMBRANE_SUBTRACTION_DIR}/run.py"  -dp . -id ${MRC_DIR} -md "labels" --save_as_mat --mrc -ds $3
echo "Subtracted masks from original micrographs and saved results in $PWD/reconstructions"
conda deactivate

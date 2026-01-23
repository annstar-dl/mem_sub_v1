#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

# Path to the membrane detection project directory
MEMBRANE_DETECTION_DIR="/home/astar/Projects/membrane_detection"
# Path to the pretrained segmentation model directory
#SEGMENTATION_DIR="/home/astar/Projects/membrane_detection/out/unet_membrane_256x256_500000"
SEGMENTATION_DIR="${MEMBRANE_DETECTION_DIR}/out/mem_mad_2025_sep+dec_warmup_200000/result"
MEMBRANE_SUBTRACTION_DIR="/home/astar/Projects/VesicleProjection" # Path to the VesicleProjection project directory
INPUT_FILE_FORMAT="mrc"  # Change to "tif" if input files are in TIFF format
SEGMENTATION_MODEL_FORMAT="onnx" # "onnx" or "paddleseg"
PADDLESEG_DIR="/home/astar/Projects/PaddleSeg" # Path to the PaddleSeg project directory
# Check if the directory argument is provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <output_directory_path> <input_files_directory>"
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
conda run -n ves_seg python "${MEMBRANE_SUBTRACTION_DIR}/mrc2image.py" "$PWD/${MRC_DIR}" -o "$PWD" --format "jpg" -dsa --scale
echo "Converted MRC files to JPG"

DS_MICROGRAPHS_PATH="$PWD/images_jpg/${MRC_DIR}_ds"
if [ ${SEGMENTATION_MODEL_FORMAT} == "onnx" ]; then
    echo "Using ONNX segmentation model format."
    conda run -n ves_seg python "${MEMBRANE_SUBTRACTION_DIR}/seg_onnx.py" \
    --model_dir "${SEGMENTATION_DIR}" \
    --onnx_fname model.onnx \
    --data_path ${DS_MICROGRAPHS_PATH} \
    --save_dir "${PWD}"
elif [ ${SEGMENTATION_MODEL_FORMAT} == "paddleseg" ]; then
    echo "Using PaddleSeg segmentation model format."
    # Perform segmentation using PaddleSeg
    conda run -n paddleseg python "${PADDLESEG_DIR}/deploy/python/infer.py" \
    --config "${SEGMENTATION_DIR}/deploy.yaml" \
    --image_path ${DS_MICROGRAPHS_PATH} \
    --save_dir "$PWD/labels_pseudcolor" \
    # convert the pseudo-color masks to binary masks
    conda run -n ves_seg python "${MEMBRANE_SUBTRACTION_DIR}/color2binary.py" "$PWD/labels_pseudcolor"

else
    echo "Invalid segmentation model format specified. Use 'onnx' or 'paddleseg'."
    exit 1
fi
echo "Converted pseudo-color masks to binary masks in $PWD/labels"

# Subtract the predicted masks from the original micrographs
conda run -n ves_seg python "${MEMBRANE_SUBTRACTION_DIR}/run_mrc_subtraction.py" \
 -dp ${PWD} -ip "$PWD/${MRC_DIR}" \
 --out_format "mrc" "png" \
 --save_reconstruction
echo "Subtracted masks from original micrographs and saved results in $PWD/reconstructions"



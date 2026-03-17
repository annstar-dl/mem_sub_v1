#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

# Path to the membrane detection project directory
MEMBRANE_DETECTION_DIR="/home/astar/Projects/membrane_detection"
# Path to the pretrained segmentation model directory
#SEGMENTATION_DIR="${MEMBRANE_DETECTION_DIR}/out/unet_membrane_256x256_500000/result"
SEGMENTATION_DIR="${MEMBRANE_DETECTION_DIR}/out/mem_mad_2025_sep+dec_warmup_200000/result"
MEMBRANE_SUBTRACTION_DIR="/home/astar/Projects/membrane_subtraction_v1/" # Path to the VesicleProjection project directory
INPUT_FILE_FORMAT="mrc"  # Change to "tif" if input files are in TIFF format
SEGMENTATION_MODEL_FORMAT="onnx" # "onnx" or "paddleseg"
PADDLESEG_DIR="/home/astar/Projects/PaddleSeg" # Path to the PaddleSeg project directory
# Check if the directory argument is provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_files_directory> <output_directory_path>"
    exit 1
fi
MRC_DIR=$(basename "$1")
# create a new directory to store the results
mkdir -p $2
#go into the new directory
cd $2
mkdir -p "misc"
# copy input files directory to an output directory
cp -r $1 "$PWD/misc/"
echo "Copied MRC files from $1 to $PWD/misc/$MRC_DIR"
# create a jpg directory to store the converted images
# mkdir -p "images_jpg"
# create a directory to store predicted masks
# mkdir -p "labels"

# Convert mrc files to jpg for segmentation
conda run -n ves_seg python "${MEMBRANE_SUBTRACTION_DIR}/tools/mrc2image.py" "$PWD/misc/${MRC_DIR}" -o "$PWD/misc" --format "jpg" -dsa --scale
echo "Converted MRC files to JPG"

DS_MICROGRAPHS_PATH="$PWD/misc/${MRC_DIR}_jpg_ds"
if [ ${SEGMENTATION_MODEL_FORMAT} == "onnx" ]; then
    echo "Using ONNX segmentation model format."
    conda run -n ves_seg python "${MEMBRANE_SUBTRACTION_DIR}/membrane_seg/seg_onnx.py" \
    --model_dir "${SEGMENTATION_DIR}" \
    --onnx_fname model.onnx \
    --data_path ${DS_MICROGRAPHS_PATH} \
    --save_dir "${PWD}/misc"
elif [ ${SEGMENTATION_MODEL_FORMAT} == "paddleseg" ]; then
    echo "Using PaddleSeg segmentation model format."
    # Perform segmentation using PaddleSeg
    conda run -n paddleseg python "${PADDLESEG_DIR}/deploy/python/infer.py" \
    --config "${SEGMENTATION_DIR}/deploy.yaml" \
    --image_path ${DS_MICROGRAPHS_PATH} \
    --save_dir "$PWD/labels_pseudcolor" \
    # convert the pseudo-color masks to binary masks
    conda run -n ves_seg python "${MEMBRANE_SUBTRACTION_DIR}/color2binary.py" "$PWD/labels_pseudcolor"
    echo "Converted pseudo-color masks to binary masks in $PWD/labels"
else
    echo "Invalid segmentation model format specified. Use 'onnx' or 'paddleseg'."
    exit 1
fi


# Subtract the predicted masks from the original micrographs
conda run -n ves_seg python "${MEMBRANE_SUBTRACTION_DIR}/tools/run_mrc_subtraction.py" \
 -dp ${PWD} -ip "$PWD/misc/${MRC_DIR}" \
 --out_format_sub "mrc" "png" \
 --out_format_mem "mrc" "png" \
 --save_angle \
 --save_subtraction

echo "Subtracted masks from original micrographs and saved results in $PWD"
# Convert mrc files to jpg
conda run -n ves_seg python "${MEMBRANE_SUBTRACTION_DIR}/tools/mrc2image.py" "$PWD/misc/membranes" -o "$PWD/misc/membranes_ds" --format "png" -dsa --scale
# Convert mrc files to jpg
conda run -n ves_seg python "${MEMBRANE_SUBTRACTION_DIR}/tools/mrc2image.py" "$PWD/subtracted_mrc" -o "$PWD/misc/subtracted_ds" --format "png" -dsa --scale


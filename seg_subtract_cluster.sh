#!/bin/bash

#SBATCH --job-name=seg_subtract
#SBATCH --output=logs/seg_subtract_%j.log
#SBATCH --error=logs/seg_subtract_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --partition=gpu_devel
#SBATCH --mem=5G
module reset
module load miniconda
conda activate ves_seg
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
INPUTDIR="/vast/palmer/scratch/sigworth/fjs2/20240703-Kv-vesicles/MicrographsU"
SAVEDIR="/home/as4873/project/vesicle_data/20240703-Kv-vesicles"
FILENAME="slot5_A28-4_63_0006_X-1Y+0-0"
# Path to the membrane detection project directory
MEMBRANE_DETECTION_DIR="/home/as4873/project/membrane_detection"
# Path to the pretrained segmentation model directory
SEGMENTATION_DIR="/home/as4873/project/membrane_detection/out/unet_membrane_256x256_500000"
# Path to the VesicleProjection project directory
MEMBRANE_SUBTRACTION_DIR="/home/as4873/project/VesicleProjection"
SEGMENTATION_MODEL_FORMAT="onnx" # "onnx" or "paddleseg"
PADDLESEG_DIR="/home/astar/Projects/PaddleSeg" # Path to the PaddleSeg project directory
# Check if the directory argument is provided
MRC_DIR=$(basename "$INPUTDIR")
# create a new directory to store the results
mkdir -p $SAVEDIR
#go into the new directory
cd $SAVEDIR
# copy input files directory to an output directory
cp -r $INPUTDIR .
# create a jpg directory to store the converted images
mkdir -p "images_jpg"
# create a directory to store predicted masks
mkdir -p "labels"
# Convert mrc files to jpg for segmentation
python "${MEMBRANE_SUBTRACTION_DIR}/mrc2image.py" "$PWD/${MRC_DIR}" \
                            -o "$PWD" --format "jpg" -dsa --scale -fn "${FILENAME}.mrc"
DS_MICROGRAPHS_PATH="$PWD/images_jpg/${MRC_DIR}_ds"
if [ ${SEGMENTATION_MODEL_FORMAT} == "onnx" ]; then
    python "${MEMBRANE_SUBTRACTION_DIR}/seg_onnx.py" \
    --model_dir "${SEGMENTATION_DIR}/result" \
    --onnx_fname model.onnx \
    --data_path ${DS_MICROGRAPHS_PATH} \
    --save_dir "${PWD}" \
    -fn "${FILENAME}.jpg"
elif [ ${SEGMENTATION_MODEL_FORMAT} == "paddleseg" ]; then
    # Perform segmentation using PaddleSeg
    conda run -n paddleseg python "${PADDLESEG_DIR}/deploy/python/infer.py" \
    --config "${SEGMENTATION_DIR}/result/deploy.yaml" \
    --image_path ${DS_MICROGRAPHS_PATH} \
    --save_dir "$PWD/labels_pseudcolor" \
    # convert the pseudo-color masks to binary masks
    python "${MEMBRANE_SUBTRACTION_DIR}/color2binary.py" "$PWD/labels_pseudcolor"

else
    echo "Invalid segmentation model format specified. Use 'onnx' or 'paddleseg'."
    exit 1
fi
# Subtract the predicted masks from the original micrographs
python "${MEMBRANE_SUBTRACTION_DIR}/run_mrc_subtraction.py" \
 -dp ${PWD} -ip "$PWD/${MRC_DIR}" \
 --out_format "mrc" "png" \
 -fn "${FILENAME}.mrc"
echo "Subtracted masks from original micrographs and saved results in $PWD/reconstructions/subtracted_png/${FILENAME}"





#!/bin/bash
MAIN_DIR="/home/astar/Projects/membrane_detection/data/membrane"
IMAGE_DIR="${MAIN_DIR}/images/test"
MASK_DIR="${MAIN_DIR}/labels/test"
noise_level=25

MODEL="swinir"
MODEL_PATH="/home/astar/Projects/SwinIR/experiments/pretrained_models/004_grayDN_DFWB_s128w8_SwinIR-M_noise${noise_level}.pth"
if [ ${MODEL} == "swinir" ]; then
    echo "Using SwinIR model."
    OUTPUT_DIR="/home/astar/Projects/vesicles_data/iclr_experiments/swinir_${noise_level}noise"
elif [ ${MODEL} == "bm3d" ]; then
    echo "Using bm3d model."
    OUTPUT_DIR="/home/astar/Projects/vesicles_data/iclr_experiments/bm3d"
else
    echo "Invalid model specified. Use 'swinir' or 'bm3d'."
    exit 1
fi


python denoise.py --input_dir $IMAGE_DIR --mask_dir $MASK_DIR --output_dir $OUTPUT_DIR --model $MODEL \
--model_path $MODEL_PATH

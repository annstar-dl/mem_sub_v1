#!/bin/bash
IMGPATH=$1
SAVEDIR=$2
SAVE_ANGLE=1
SAVE_SUB=1
SEGMENTATION_DIR="/home/astar/Projects/membrane_detection/out/mem_mad_2026_march_warmup_lr0012_200000/result"
export SAVEDIR=$SAVEDIR
export SEGMENTATION_DIR=$SEGMENTATION_DIR
export SAVE_ANGLE=$SAVE_ANGLE
export SAVE_SUB=$SAVE_SUB
bash scripts/seg_subtract_v1.sh "${IMGPATH}"
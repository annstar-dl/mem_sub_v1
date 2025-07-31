#!/bin/bash
# This script runs the Python script with the specified arguments.
DATAPATH="/home/astar/Projects/vesicles_data/new_data"
python run.py --dataset_path $DATAPATH --imgs_dir "images" \
--masks_dir "labels" --save_as_mat
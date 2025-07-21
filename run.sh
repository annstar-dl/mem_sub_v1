#!/bin/bash
# This script runs the Python script with the specified arguments.
IMGSPATH="/home/astar/Projects/vesicles_data/images_fred_mck"
python run.py --imgs_path $IMGSPATH \
--masks_dir "labels_fred_mck" --flatten_bg --save_as_mat
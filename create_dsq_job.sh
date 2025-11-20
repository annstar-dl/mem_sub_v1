#!/bin/bash
#create a job_list for dsq submission
#Note: slurm jobs start in the directory from which your job was submitted.
DATASET_PATH=$1
SEG_MODEL_PATH=$2
SAVE_DIR_PATH=$3
module load miniconda
module load dSQ
conda activate ves_seg
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python create_job_list.py -ddp ${DATASET_PATH} -jfp ./joblist.txt \
    -segmp ${SEG_MODEL_PATH} -savedp ${SAVE_DIR_PATH}
# the joblist.txt will be created in the current directory
# Now create the dsq job submission script
dsq --job-file joblist.txt --mem=5G --cpus-per-task=5 --gres=gpu:1 -t 11:00 --partition=scavenge_gpu --mail-type ALL
#--output=/dev/null
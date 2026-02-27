#!/bin/bash
#create a job_list for dsq submission
#Note: slurm jobs start in the directory from which your job was submitted.
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 DATASET_PATH SEG_MODEL_PATH SAVE_DIR_PATH JOB_ARRAY_NAME SAVE_ANGLE SAVE_SUB"
    exit 1
fi
DATASET_PATH=$1
SEG_MODEL_PATH=$2
SAVE_DIR_PATH=$3
JOB_ARRAY_NAME=$4
SAVE_ANGLE=$5
SAVE_SUB=$6
SHOW_OUTPUT=$7
TIMESTEMP=$(date +"%Y%m%d_%H%M%S")
module load miniconda
module load dSQ
conda activate ves_seg
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python scripts/create_job_list.py -ddp ${DATASET_PATH} -jfp "./joblist.txt" \
    -segmp ${SEG_MODEL_PATH} -savedp ${SAVE_DIR_PATH} \
    --save_angle_flag=${SAVE_ANGLE} --save_sub_flag=${SAVE_SUB} \
# the joblist.txt will be created in the current directory
# Now create the dsq job submission script
if [ "$SHOW_OUTPUT" -eq 1 ]; then
    dsq --job-file joblist.txt --mem=5G --cpus-per-task=4 --gres=gpu:1 -t 15:00 --partition=scavenge_gpu --mail-type ALL  --batch-file="${JOB_ARRAY_NAME}_${TIMESTEMP}_dsq_job.sh"
else
    dsq --job-file joblist.txt --mem=5G --cpus-per-task=4 --gres=gpu:1 -t 15:00 --partition=scavenge_gpu --mail-type ALL  --batch-file="${JOB_ARRAY_NAME}_${TIMESTEMP}_dsq_job.sh" --output=/dev/null
fi
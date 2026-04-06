#!/bin/bash
#create a job_list for dsq submission
#Note: slurm jobs start in the directory from which your job was submitted.
if [ "$#" -lt 5 ]; then
    echo "Usage: $0 DATASET_PATH SAVE_DIR_PATH JOB_ARRAY_NAME SAVE_ANGLE SAVE_SUB [show_output] [nb_of_jobs]"
    exit 1
fi
DATASET_PATH=$1
SAVE_DIR_PATH=$2
JOB_ARRAY_NAME=$3
SAVE_ANGLE=$4
SAVE_SUB=$5
nb_of_jobs="${6:--1}"
show_output="${7:-0}"

TIMESTEMP=$(date +"%Y%m%d_%H%M%S")
module load miniconda
module load dSQ
conda activate ves_seg
# the joblist.txt will be created in the current directory
python scripts/create_job_list.py -ddp ${DATASET_PATH} -jfp "./joblist.txt" \
    -savedp ${SAVE_DIR_PATH} --save_angle_flag=${SAVE_ANGLE} \
    --save_sub_flag=${SAVE_SUB} \
    --nb_of_jobs=${nb_of_jobs}
# check if jobfile is not empty
if [ -s joblist.txt ]; then
  #Now create the dsq job submission script
  if [[ "$SHOW_OUTPUT" -eq 1 ]]; then
      dsq --job-file joblist.txt --mem=5G --cpus-per-task=4 --gpus=1 -t 20:00 --partition=scavenge_gpu --mail-type ALL  --batch-file="${JOB_ARRAY_NAME}_${TIMESTEMP}_dsq_job.sh" --output=/dev/null
  else
      dsq --job-file joblist.txt --mem=5G --cpus-per-task=4 --gpus=1 -t 20:00 --partition=scavenge_gpu --mail-type ALL  --batch-file="${JOB_ARRAY_NAME}_${TIMESTEMP}_dsq_job.sh"
  fi
  else
      echo "Error: joblist.txt is empty. No jobs to submit."
fi

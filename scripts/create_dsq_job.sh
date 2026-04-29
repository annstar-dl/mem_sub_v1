#!/bin/bash
#create a job_list for dsq submission
#Note: slurm jobs start in the directory from which your job was submitted.
if [ "$#" -lt 5 ]; then
    echo "Usage: $0 DATASET_PATH SAVE_DIR_PATH JOB_ARRAY_NAME SAVE_ANGLE SAVE_SUB [show_output] [nb_of_jobs] [seg_dir]"
    exit 1
fi
DATASET_PATH=$1
SAVE_DIR_PATH=$2
JOB_ARRAY_NAME=$3
SAVE_ANGLE=$4
SAVE_SUB=$5
show_output="${6:-0}"
nb_of_jobs="${7:--1}"
seg_dir="${8:-""}"
#copy the yml parameter file in the same directory as results add a timestamp to avoid overwriting previous results
if [[ -f "${SAVE_DIR_PATH}" ]]; then
  echo "The directory ${SAVE_DIR_PATH} already exists. Please choose a different directory or remove the existing one."
  exit 1
fi
mkdir -p "${SAVE_DIR_PATH}"
cp "parameters.yml" "${SAVE_DIR_PATH}/parameters.yml"

TIMESTEMP=$(date +"%Y%m%d_%H%M%S")
module load miniconda
module load dSQ
conda activate ves_seg
# the joblist.txt will be created in the current directory
python scripts/create_job_list.py -ddp ${DATASET_PATH} -jfp "./joblist.txt" \
    -savedp ${SAVE_DIR_PATH} --save_angle_flag=${SAVE_ANGLE} \
    --save_sub_flag=${SAVE_SUB} \
    --nb_of_jobs=${nb_of_jobs} \
    --seg_dir_path="${seg_dir}"
# check if jobfile is not empty
if [ -s joblist.txt ]; then
  #Now create the dsq job submission script
  if [[ "$show_output" -eq 1 ]]; then
    echo "Submitting jobs with output shown in the terminal."
      dsq --job-file joblist.txt --mem=5G --cpus-per-task=4 --gpus=1 -t 20:00 --partition=scavenge_gpu --mail-type ALL  --batch-file="${JOB_ARRAY_NAME}_${TIMESTEMP}_dsq_job.sh"
  else
      dsq --job-file joblist.txt --mem=5G --cpus-per-task=4 --gpus=1 -t 20:00 --partition=scavenge_gpu --mail-type ALL  --batch-file="${JOB_ARRAY_NAME}_${TIMESTEMP}_dsq_job.sh" --output=/dev/null
  fi
  else
      echo "Error: joblist.txt is empty. No jobs to submit."
fi

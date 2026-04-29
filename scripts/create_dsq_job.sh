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
#timestep to add to the job array name to avoid overwriting previous results, this is useful for later reference and to avoid confusion about which code was used for segmentation
TIMESTEMP=$(date +"%Y%m%d_%H%M%S")
#copy the yml parameter file in the same directory as results add a timestamp to avoid overwriting previous results
mkdir -p "${SAVE_DIR_PATH}"
if [[ -f "${SAVE_DIR_PATH}/parameters.yml" ]]; then
  #chekc if "${SAVE_DIR_PATH}/parameters.yml" is the same as "parameters.yml" in the current directory, if not copy the new one
  #if not copy the exit the code and request to fix the parameters.yml file
  #in the future change this to use old "${SAVE_DIR_PATH}/parameters.yml" as parameters.yml file, when the path to that file becomes an argument
  if ! cmp -s "parameters.yml" "${SAVE_DIR_PATH}/parameters.yml"; then
    echo "Error: ${SAVE_DIR_PATH}/parameters.yml already exists and is different from the current parameters.yml. Please fix this before running the script."
    exit 1
  else
    echo "parameters.yml already exists in ${SAVE_DIR_PATH} and is the same as the current parameters.yml. No need to copy."
  fi
else
  cp "parameters.yml" "${SAVE_DIR_PATH}/parameters.yml"
fi



module load miniconda
module load dSQ
conda activate ves_seg

if [[ ! -f "${SAVE_DIR_PATH}/exp_config.yml" ]]; then
  python "tools/record_hash.py" -sp "${SAVE_DIR_PATH}"
else
  python "tools/record_hash.py" -sp "${SAVE_DIR_PATH}" --compare_metadata
fi

mkdir -p "dsq_files"

# the joblist.txt will be created in the current directory
python scripts/create_job_list.py -ddp ${DATASET_PATH} -jfp "dsq_files/joblist_${TIMESTEMP}.txt" \
    -savedp ${SAVE_DIR_PATH} --save_angle_flag=${SAVE_ANGLE} \
    --save_sub_flag=${SAVE_SUB} \
    --nb_of_jobs=${nb_of_jobs} \
    --seg_dir_path="${seg_dir}"

# check if jobfile is not empty
if [ -s joblist.txt ]; then
  #Now create the dsq job submission script
  if [[ "$show_output" -eq 1 ]]; then
    echo "Submitting jobs with output shown in the terminal."
      dsq --job-file "dsq_files/joblist_${TIMESTEMP}.txt" --mem=5G --cpus-per-task=4 --gpus=1 -t 20:00 --partition=scavenge_gpu --mail-type ALL  --batch-file="dsq_files/${JOB_ARRAY_NAME}_${TIMESTEMP}_dsq_job.sh" --status-file dsq_files/status.tsv
  else
      dsq --job-file "dsq_files/joblist_${TIMESTEMP}.txt" --mem=5G --cpus-per-task=4 --gpus=1 -t 20:00 --partition=scavenge_gpu --mail-type ALL  --batch-file="dsq_files/${JOB_ARRAY_NAME}_${TIMESTEMP}_dsq_job.sh" --status-file dsq_files/status.tsv --output dsq_files/slurm-%A_%a.out
  fi
  else
      echo "Error: joblist.txt is empty. No jobs to submit."
fi


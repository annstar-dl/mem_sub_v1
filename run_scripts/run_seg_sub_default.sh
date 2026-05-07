#!/bin/bash
set -euo pipefail


DATASET_PATH=""#name of the dataset to be processed, this can be a single mrc file or a directory containing mrc files
SAVE_DIR_PATH=""#directory where the results will be saved
JOB_ARRAY_NAME=""#if you want to run dsq job submission, this will be a prefix to the name of the job array
SAVE_ANGLE=0 #flag to indicate whether to save the angle of the predicted membrane in the output, this is useful for later analysis and visualization
SAVE_SUB=1 #flag to indicate whether to save the subtracted micrographs in the output, this is useful for later analysis and visualization
SHOW_OUTPUT=0 #flag to indicate whether to show the output of the dsq jobs in the terminal, this is useful for debugging and monitoring the progress of the jobs
NB_OF_JOBS=-1 #number of jobs to submit to the cluster, if -1, all jobs will be submitted, this is useful for testing and debugging
SEGMENTATION_DIR="membrane_seg/seg_model/mem_mad_2026_march_warmup_lr_0005_500000" #path to the segmentation model, this should be a directory containing the model.onnx file, this is useful for later reference and to avoid confusion about which model was used for segmentation
PAR_FPATH="parameters.yml" #path to the parameters.yml file, this is useful for later reference and to avoid confusion about which parameters were used for segmentation
RUN_DSQ=1 #flag to indicate whether to run the dsq job submission script. If use HPC set it to 1, if 0, the whole folder will be processed at once using the seg_subtract_desktop.sh script
RUN_ONLY_SEGMENTATION=0 #flag to indicate whether to run only the segmentation script, this is useful for testing and debugging. Always keep it zero.

mkdir -p "$SAVEDIR"

#record parameters.yml in the same directory as results, this is useful for later reference and to avoid confusion about which parameters were used for segmentation, add a timestamp to avoid overwriting previous results
if [[ -f "${SAVEDIR}/parameters.yml" ]]; then
  #chekc if ${PAR_FPATH} is the same as parameters.yml in the current directory
  if ! cmp -s ${PAR_FPATH} "${SAVEDIR}/parameters.yml"; then
    echo "Error: ${par_fname} already exists and is different from the current parameters.yml. Please fix this before running the script."
    exit 1
  else
    echo "parameters file already exists in ${SAVEDIR} and is the same as the current parameters.yml. No need to copy."
  fi
  else
  cp ${PAR_FPATH} "${SAVEDIR}/parameters.yml"
fi

#record the current commit hash in a yml file in the savedir if it doesn't already exist, this is useful for later reference and to avoid confusion
python -m mem_sub.record_hash -sp "${SAVEDIR}"

#save the segmentation model path in a text file in the savedir if it doesn't already exist, this is useful for later reference and to avoid confusion about which model was used for segmentation
if [[ ! -f "${SAVEDIR}/seg_model.txt" ]]; then
  echo "segmentation model: ${SEGMENTATION_DIR}" > "${SAVEDIR}/seg_model.txt"
else
    expected_content="segmentation model: ${SEGMENTATION_DIR}"
    existing_content=$(cat "${SAVEDIR}/seg_model.txt")
  if [[ "$expected_content" != "$existing_content" ]]; then
    echo "Error: ${SAVEDIR}/seg_model.txt already exists and is different from the current segmentation model. Please fix this before running the script."
    exit 1
  else
    echo "seg_model.txt already exists in ${SAVEDIR} and is the same as the current segmentation model. No need to copy."
  fi
fi


#run the dsq job submission script if the flag is set, this will create a job list and submit the jobs to the cluster, otherwise run the whole folder at once using the seg_subtract_desktop.sh script
if [[ "${RUN_DSQ}" -eq 1 ]]; then
  echo "Run dsq job submission script"
  bash scripts/create_dsq_job.sh "${DATASET_PATH}" "${SAVE_DIR_PATH}" "${JOB_ARRAY_NAME}" "${SAVE_ANGLE}" "${SAVE_SUB}" "${SHOW_OUTPUT}" "${NB_OF_JOBS}" "${SEGMENTATION_DIR}" "${PAR_FPATH}"
else
  #run only the segmentation script if the flag is set, this is useful for testing and debugging
  if [[ "${RUN_ONLY_SEGMENTATION}" -eq 1 ]]; then
    echo "Run only segmentation script"
    bash scripts/seg_mrc.sh "${DATASET_PATH}" "${SAVE_DIR_PATH}" "${SEGMENTATION_DIR}" "${PAR_FPATH}"
    exit 0
    fi
  echo "Run subtraction on a whole folder at once"
  bash scripts/seg_subtract_desktop.sh "${DATASET_PATH}" "${SAVE_DIR_PATH}" "${SEGMENTATION_DIR}" "${PAR_FPATH}"
fi
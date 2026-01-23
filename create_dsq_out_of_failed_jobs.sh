#!/bin/bash
# This script creates a dSQ job submission for processing failed jobs from a previous dSQ run
JOB_ARRAY_ID=$1
module load dSQ
dsqa -j ${JOB_ARRAY_ID} -f joblist.txt -s BOOT_FAIL,CANCELLED,DEADLINE,FAILED,NODE_FAIL,OUT_OF_MEMORY,PREEMPTED,TIMEOUT > re-run_jobs.txt 2> ${JOB_ARRAY_ID}_report.txt
dsq --job-file re-run_jobs.txt --mem=10G --cpus-per-task=5 --gres=gpu:1 -t 20:00 --partition=scavenge_gpu --mail-type ALL --output=/dev/null
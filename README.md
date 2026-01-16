# Membrane Subtraction
This repository contains code and resources for performing membrane subtraction in Cryo-EM imaging data. Membrane subtraction is a technique used to enhance the visibility of protein structures by removing membrane outlines.
## Contents
- [Features](#features)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
### Features
- Automated membrane outline detection
- High-performance subtraction algorithms
- Support for mrc file formats
- Integration with Yale HPC resources
### Requirements
- Python 3.10 or higher
- Conda package manager
- GPU with CUDA support (optional but recommended for performance)
- Packages listed in `environment.yml`
### Usage
1. Clone the repository:
```bash
git clone 
#go to the repository directory
cd VesicleProjection
```
2. Installation of dependencies:
```bash
# Create a new environment from the YAML file
conda env_name create -f environment.yml
conda activate mem_sub
```
3. Membrane subtraction on a folder of mrc files:
```bash
   bash seg_subtract.sh /path/to/save/results /path/to/mrc/files 
```
4. HPC Usage. Membrane subtraction on Yale HPC cluster is done using Deadly Simple Queue (DSQ) scheduler.
   The idea is that every image can be processed independently, so we can submit many jobs to the cluster,
each processing a single image. This way we can "scavenge" free GPU resources from other users.
    To run DSQ we have to prepare file with the list of jobs and their parameters. 
However, Yale HPC has a limit on how short the job duration can be, 
so we have to batch several image processing jobs into one. 
After creating the job file, you can DSQ job using following script:
```bash
   bash create_dsq_jobs.sh /path/to/job/file.txt /path/to/conda/env/mem_sub
```
When running on Yale HPC, make sure to submit the job using the appropriate scheduler commands.
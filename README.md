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
This files contains two main parts:
- Segmentation of membrane outlines using pretrained U-Net model.
- Subtraction of the segmented membrane outlines from the original images.

The first step requires downloading U-Net model weights. These weights can be downloaded from:
[U-Net Weights Download Link](https://example.com/unet_weights.pth)
Make sure to place the downloaded weights in the appropriate directory and specify the path to model.onnx file in the `seg_subtract.sh` script in SEGMENTATION_DIR variable.
Also before the segmentation step, the mrc files are downsapled to have voxel size of 4.5 Angstroms. As a result, you will see in the output folder images_jpg/{input_folder_name}_ds with downsampled images in jpg format.
The structure of the output folder will be as follows:
```/your/save/path/
    |---subtractions_mrc/ # Images after membrane subtraction in mrc format
    |---misc/ # Miscellaneous files, including logs and intermediate results
    ├──---images_jpg/{input_folder_name}_ds/  # Downsampled images in jpg format
    ├──---labels/                     # Segmented membrane masks
    ├──---input_mrc_folder_name/      # Original mrc files
    ├──---reconstructions/  
    ├─────subtractions_png/ # Images after membrane subtraction in png format
    ├─────reconstructed_membranes/ # Reconstracted membrane images
```
Additionally original mrc files are copied to the output folder for convenience. This could be prevented by commenting out "cp -r $2 ." line in the `seg_subtract.sh` script.
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
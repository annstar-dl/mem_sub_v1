# Membrane Subtraction
This repository contains code and resources for performing membrane subtraction in Cryo-EM imaging data. Membrane subtraction is a technique used to enhance the visibility of protein structures by removing membrane outlines.
This project is developed by Tagare lab, at the Radiology and Biomedical Imaging department at Yale University.
For any questions or issues, please contact the Anna Starynska (anna.starynska@yale.edu) or Hemant Tagare (hemant.tagare@yale.edu).
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
git clone https://github.com/annstar-dl/mem_sub_v1
#go to the repository directory
cd mem_sub_v1
```
2. Installation of dependencies and environment setup:
    1. Create a new conda environment and install the required packages using the provided `environment.yml` file:
        
        - For desktop or local machine usage:
            ```bash
            # Create a new environment from the YAML file
            conda env_name create -f environment.yml
            conda activate mem_sub
            pip install -e .
            ```
       - For HPC usage, please load the miniconda module first and then create the environment:
            ```bash
            # Load the miniconda module
            module load miniconda
            # Create a new environment from the YAML file
            conda env_name create -f environment.yml
            conda activate mem_sub
            pip install -e .
            ```
    2. Download the pretrained U-Net model weights, and file with preprocessing parameters from the provided link and place them in the appropriate directory. 
    From the provided link, download folder with .onnx and .yml files and place this directory in the membrane_seg/seg_model/ directory.
   For example, if you downloaded the folder with name "mem_mad_2026_march_warmup_lr0012_200000". The membrane_seg/seg_model/ directory should look like this:
   ```
   membrane_seg/seg_model/
    |---mem_mad_2026_march_warmup_lr0012_200000/
    |------model.onnx
    |------deploy.yaml
    ```
3. Run the subtraction script:

     !!!Make sure to place the downloaded weights in the appropriate directory and specify the path to model directory in the `script/seg_mrc.sh` script in SEGMENTATION_DIR variable.
   1. Membrane subtraction for a folder of mrc file on a desktop or local machine can be done using the following script:
    ```bash
       bash script/seg_subtract_destop.sh /path/to/mrc/files /path/to/save/results
    ```

    2. For HPC usage, please refer to the HPC Usage section below.
4. HPC Usage. Membrane subtraction on Yale HPC cluster is done using Deadly Simple Queue (DSQ) scheduler.
   The idea is that every image can be processed independently, so we can submit many jobs to the cluster,
each processing a single image. To run DSQ we have to prepare file with the list of jobs and their parameters. 
However, Yale HPC has a limit on how short the job duration can be, 
so we have to batch several image processing jobs into one. 
You can create the job file, you can run DSQ job using following script:
```bash
   bash script/create_dsq_jobs.sh /path/to/dataset/folder /path/to/save/results/folder job_array_name save_angle save_sub show_output
```
Here is the explanation of the parameters:
- `/path/to/dataset/folder`: Path to the folder containing mrc files to be processed.
- `/path/to/save/results/folder`: Path to the folder where the results will be saved.
- `job_array_name`: A name for the job array, which will be used to identify the jobs in the DSQ scheduler, e.g. liposome.
- `save_angle`: A flag (0 or 1) indicating whether to save the angle information of the segmented membranes.
- `save_sub`: A flag (0 or 1) indicating whether to save the subtracted images.
- `nb_of_jobs`: (optinal argument) How many jobs to run. If you want to process all the micrographs set do not set it at all. 
                For the test run set it to 1.
- `show_output`: (optinal argument) A flag (0 or 1) indicating whether to print the output of the DSQ jobs to the log files.
                Default is 0, which means that the output will not be printed to the log files. 

After running script/create_dsq_jobs.sh it will print out a line:
- `To submit the job array, run: sbatch liposome_12345678.sh`
Paste this command into the terminal to submit the job array to the DSQ scheduler.
5. Results and Output Structure. After running the subtraction script, you will find the results in the specified output folder.
This files contains two main parts:
- Segmentation of membrane outlines using pretrained U-Net model.
- Subtraction of the segmented membrane outlines from the original images.


Also before the segmentation step, the mrc files are downsapled to have voxel size of 4.5 Angstroms. As a result, you will see in the output folder misc/{input_folder_name}_ds with downsampled images in jpg format.
The structure of the output folder will be as follows:
```/your/save/path/
    |---subtractions_mrc/ # Images after membrane subtraction in mrc format
    |---misc/ # Miscellaneous files, including logs and intermediate results
    ├──---{input_folder_name}_ds/  # Downsampled micrographs in jpg format
    ├──---labels/                     # Segmented membrane masks (downsampled)
    ├─────subtracted_png_ds/ # Downsampled after membrane subtraction in png format
    ├─────membranes/ # Images of membrane estimates in png format
    |─────membranes_ds/ # Downsampled images of membrane estimates in png format`
```


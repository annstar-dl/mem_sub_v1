# Membrane Subtraction
This repository contains code and resources for performing membrane subtraction in Cryo-EM imaging data. Membrane subtraction is a technique used to enhance the visibility of protein structures by removing membrane outlines.
This project is developed by Tagare lab, at the Radiology and Biomedical Imaging department at Yale University.
For any questions or issues, please contact the Anna Starynska (anna.starynska@yale.edu) or Hemant Tagare (hemant.tagare@yale.edu).
## Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [First run](#first-run)
- [Change Hyperparameters](#change-hyperparameters)
- [License](#license)
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
### Installation
1. Clone the repository. Set the branch name to latest release version (you can find the list of release tags on the right-hand side.) 
You can clone the repository using the following command (replace v1.0.0 with your actual release tag):
```bash
  git clone --branch v1.0.0 --depth 1 https://github.com/annstar-dl/mem_sub_v1
```
2. Installation create the conda environment and install packages.
    1. For desktop or local machine usage:
    ```bash
        #go to the repository directory
        cd mem_sub_v1
        # Create a new environment from the YAML file
        # it will create a new conda environment named ves_seg and install all the required packages listed in the environment.yml file
        conda env create -f environment.yml
        conda activate ves_seg
        #install our package mem_sub in editable mode
        pip install -e .
    ```
   
   2. For HPC usage, please load the miniconda module first and then create the environment:

       REMINDER: Switch from the login node while installing the environment to compute node, otherwise installation may fail due to limited resources. 
        ```bash
          #this command will allocate a compute node for you, make sure to specify the resources you need, e.g. number of GPUs, memory, etc. For example:
          salloc
        ```
         After you are on the compute node, run the following commands to create the environment:
    
        ```bash
          # Load the miniconda module
          module load miniconda
          # Create a new environment from the YAML file
          conda env create -f environment.yml
          conda activate ves_seg
          pip install -e .
        ```
3. Download the pretrained U-Net model weights, and file with preprocessing parameters from the provided link (link is provided by request) and place them in the appropriate directory. 
From the provided link, download folder with .onnx and .yml files and place this directory in the membrane_seg/seg_model/ directory.
   For example, if you downloaded the folder with name "mem_mad_2026_march_warmup_lr0012_200000". The membrane_seg/seg_model/ directory should look like this:
   Reminder: do not rename the model directory! At this stage we are tracking model name for experiments reproducibility.
    ```
   membrane_seg/seg_model/
    |---mem_mad_2026_march_warmup_lr0012_200000/
    |------model.onnx
    |------deploy.yaml
    ```
4. Copy the default subtraction parameters yml file (parameters_default.yml) from the repository to the desired location, and modify it if needed. If you do not want to pass this file path as an argument when running the subtraction script, put this file in the project root and name it parameters.yml.
    ```bash
   #go to the repository directory if you are not there already
   cd mem_sub_v1
   #copy the default parameters yml file to the repository root
   cp parameters_default.yml parameters.yml
    ```

### Usage
1. Run the membrane subtraction script. Depending on your computational resources and preferences, you can choose to run the subtraction on a desktop or on a high-performance computing (HPC) cluster.
   1. Navigate to run_scripts directory and make a copy of run_seg_sub_default.sh script
        ```bash
            cd run_scripts
            cp run_seg_sub_default.sh run_seg_sub.sh
        ```
   
        Open the run_seg_sub.sh script and set all the parameters to the desired values. Keep most of the parameters as they are, but make sure to set the correct paths for your dataset and output directory.
        Also change the membrane segmentation model path if you are processing non vesicle data, such as bacterial micrographs.
    
        Here is the description of the parameters:
        - DATASET_PATH - name of the dataset to be processed, this can be a single mrc file or a directory containing mrc files
        - SAVE_DIR_PATH - directory where the results will be saved
        - JOB_ARRAY_NAME - if you want to run dsq job submission, this will be a prefix to the name of the job array
        - SAVE_ANGLE - (0 or 1) flag to indicate whether to save the angle of the predicted membrane in the output, this is useful for later analysis and visualization
        - SAVE_SUB=1 - (0 or 1) flag to indicate whether to save the subtracted micrographs in the output, this is useful for later analysis and visualization
        - SHOW_OUTPUT - (0 or 1) flag to indicate whether to show the output of the dsq jobs in the terminal, this is useful for debugging and monitoring the progress of the jobs
        - NB_OF_JOBS - (default -1) number of jobs to submit to the cluster, if -1, all jobs will be submitted, this is useful for testing and debugging
        - SEGMENTATION_DIR - path to the segmentation model, this should be a directory containing the model.onnx file, this is useful for later reference and to avoid confusion about which model was used for segmentation
        - PAR_FPATH - path to the parameters.yml file, this is useful for later reference and to avoid confusion about which parameters were used for segmentation
        - RUN_DSQ - flag to indicate whether to run the dsq job submission script. If use HPC set it to 1, if 0, the whole folder will be processed at once using the seg_subtract_desktop.sh script
        - RUN_ONLY_SEGMENTATION - flag to indicate whether to run only the segmentation script, this is useful for testing and debugging. Always keep it zero.

    2. Desktop usage. Set the parameters in the run_seg_sub.sh script run_dsq=0 and run the script using the following command:
         ```bash
          bash run_scripts/run_seg_sub.sh
         ```
    3. HPC Usage. Membrane subtraction on Yale HPC cluster is done using Deadly Simple Queue (DSQ) scheduler.
       The idea is that every image can be processed independently, so we can submit many jobs to the cluster,
    each processing a single image. To run DSQ we have to prepare file with the list of jobs and their parameters. 
    However, Yale HPC has a limit on how short the job duration can be, 
    so we have to batch several image processing jobs into one. 
    You can create the DSQ job file by running following script after you set run_dsq=1 in the run_seg_sub.sh:
        ```bash
           run_scripts/run_seg_sub.sh
        ```
         After running run_scripts/run_seg_sub.sh it will print out a line:
   
             ```
                 To submit the job array, run: sbatch liposome_12345678.sh 
             ```
   
         Paste this command into the terminal to submit the job array to the DSQ scheduler.
   
2. Results and Output Structure. After running the subtraction script, you will find the results in the specified output folder.
    This files contains two main parts:
     - Segmentation of membrane outlines using pretrained U-Net model.
     - Subtraction of the segmented membrane outlines from the original images.


Also before the segmentation step, the mrc files are downsampled to have voxel size of 4.0 Angstroms. As a result, you will see in the output folder misc/{input_folder_name}_ds with downsampled images in jpg format.
The structure of the output folder will be as follows:
        
            ```/your/save/path/
                ├──exp_config.yml/  # The configuration file used for this run
                ├──parameters.yml/  # Copy of your subtraction parameters file used for this run
                ├──subtractions_mrc/ # Images after membrane subtraction in mrc format
                ├──misc/ # Miscellaneous files, including logs and intermediate results
                ├─────seg_model.txt/  # file with the name of the segmentation model used for this run
                ├─────{input_folder_name}_ds/  # Downsampled micrographs in jpg format
                ├─────labels/                     # Segmented membrane masks (downsampled)
                ├─────subtracted_png_ds/ # Downsampled after membrane subtraction in png format
                ├─────membranes/ # Images of membrane estimates in png format
                |─────membranes_ds/ # Downsampled images of membrane estimates in png format`
            ```
### First run
For the first run, we recommend testing our code on a small subset of data to ensure that everything is set up correctly. 
You do not need to create a separate set, simply run the DSQ job creating script for a single job, which will process only 10 images.
Also allow the script to print the output of the DSQ jobs to the log files, so you can see if there are any errors or issues with the processing.
For that set the parameter show_output=1 and nb_of_jobs=1 in the run_seg_sub.sh script. This will allow you to see the output of the DSQ jobs in the terminal and check if there are any errors or issues with the processing.
After the first run, check the output folder to see if the results are as expected.
If not send us the .out file with the log output of the DSQ jobs, which you can find in the dsq_files folder in root of project directory.

### Change Hyperparameters
You can change the hyperparameters of the subtraction process by modifying your `parameters.yml` file. This file contains various parameters that control the behavior of the subtraction algorithms, such as grid step, bases radius and etc.

### License
This project is licensed under the BSD 3 License. See the LICENSE file for more details.
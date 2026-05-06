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
   ```
   membrane_seg/seg_model/
    |---mem_mad_2026_march_warmup_lr0012_200000/
    |------model.onnx
    |------deploy.yaml
    ```

### Usage
1. Copy the default subtraction parameters yml file (parameters_default.yml) from the repository to the desired location, and modify it if needed. If you do not want to pass this file path as an argument when running the subtraction script, put this file in the project root and name it parameters.yml.
    ```bash
   #go to the repository directory if you are not there already
   cd mem_sub_v1
   #copy the default parameters yml file to the repository root
   cp parameters_default.yml parameters.yml
    ```
2. Download the directory with pretrained U-Net model weights (model.onnx) and file with preprocessing parameters (deploy.yaml) from the provided link (link is provided by request) and place them in the membrane_seg/seg_model directory.
Do not rename the model directory! At this stage we are tracking model name for experiments reproducibility.

    

3. Run the membrane subtraction script. Depending on your computational resources and preferences, you can choose to run the subtraction on a desktop or on a high-performance computing (HPC) cluster.
   1. Desktop usage. Membrane subtraction on a desktop or local machine can be done using the following script:
        ```bash
           bash script/seg_subtract_destop.sh /path/to/mrc/files /path/to/saved/results path/to/segmentation_model_dir path/to/subtraction_parameters_yml_file
        ```
       if you want to set the path segmentation model to default (mem_mad_2026_march_warmup_lr_50000) set this argument to empty string, like this:
        ```bash
           bash script/seg_subtract_destop.sh /path/to/mrc/files /path/to/save/results "" path/to/subtraction_parameters_yml_file
        ```
        if you want to set the path to subtraction parameters yml file to default, do not set this parameter, like this:
        ```bash
           bash script/seg_subtract_destop.sh /path/to/mrc/files /path/to/save/results ""
        ```
   2. HPC Usage. Membrane subtraction on Yale HPC cluster is done using Deadly Simple Queue (DSQ) scheduler.
      The idea is that every image can be processed independently, so we can submit many jobs to the cluster,
   each processing a single image. To run DSQ we have to prepare file with the list of jobs and their parameters. 
   However, Yale HPC has a limit on how short the job duration can be, 
   so we have to batch several image processing jobs into one. 
   You can create the job file, you can run DSQ job using following script:
       ```bash
          bash script/create_dsq_jobs.sh /path/to/dataset/folder /path/to/save/results/folder job_array_name save_angle save_sub show_output nb_of_jobs path/to/segmentation_model_dir path/to/subtraction_parameters_yml_file
       ```
        <mark>Beware that optional parameters in Bash are not named. So if you skip one optional parameter, you have to skip all the parameters that are after it. Otherwise, the next value would be taken as the value of the parameter you skipped, and it would cause an error.</mark>
        
        Here is the description of the parameters:
        - `/path/to/dataset/folder`: Path to the folder containing mrc files to be processed.
        - `/path/to/save/results/folder`: Path to the folder where the results will be saved.
        - `job_array_name`: A name for the job array, which will be used to identify the jobs in the DSQ scheduler, e.g. liposome.
        - `save_angle`: A flag (0 or 1) indicating whether to save the angle information of the segmented membranes.
        - `save_sub`: A flag (0 or 1) indicating whether to save the subtracted images.
        - `show_output`: (optinal argument) A flag (0 or 1) indicating whether to print the output of the DSQ jobs to the log files.
                        Default is 0, which means that the output will not be printed to the log files. 
        - `nb_of_jobs`: (optinal argument) How many jobs to run. Default value –1, which means process all the files. For the test run set it to 1.
        - `path/to/segmentation_model_dir`: (optinal argument) Path to the directory with segmentation model. If not set, it will be taken from the SEGMENTATION_DIR variable in the `script/seg_mrc.sh` script.
        - `path/to/subtraction_parameters_yml_file`: (optinal argument) Path to the subtraction parameters yml file. If not set, `parameters.yml` file script.

        After running script/create_dsq_jobs.sh it will print out a line:
   
            ```
                To submit the job array, run: sbatch liposome_12345678.sh 
            ```
   
        Paste this command into the terminal to submit the job array to the DSQ scheduler.

        Here is an example of how to run the script with all parameters:
        ```bash
           bash script/create_dsq_jobs.sh /path/to/dataset/folder /path/to/save/results/folder kv_protein 0 1 0 –1 membrane_seg/seg_model/mem_mad_2026_march_warmup_lr0012_200000 parameters.yml
        ```
        This command will create a job array named "kv_protein", which will process the mrc files in the specified dataset folder, save the subtracted images, but not save the angle information of the segmented membranes. The output of the DSQ jobs will not be printed to the log files.

3. Results and Output Structure. After running the subtraction script, you will find the results in the specified output folder.
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
### First run
For the first run, we recommend testing our code on a small subset of data to ensure that everything is set up correctly. 
You do not need to create a separate set, simply run the DSQ job creating script for a single job, which will process only 10 images.
Also allow the script to print the output of the DSQ jobs to the log files, so you can see if there are any errors or issues with the processing.
```bash
    bash script/create_dsq_jobs.sh /path/to/dataset/folder /path/to/save/results/folder job_array_name 1 1 1 1
```
After the first run, check the output folder to see if the results are as expected.
If not send us the .out file with the log output of the DSQ jobs, which you can find in the root of project directory.

### Change Hyperparameters
You can change the hyperparameters of the subtraction process by modifying your `parameters.yml` file. This file contains various parameters that control the behavior of the subtraction algorithms, such as grid step, bases radius and etc.

### License
This project is licensed under the BSD 3 License. See the LICENSE file for more details.
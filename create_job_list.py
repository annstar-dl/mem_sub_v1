import os
from typing import Iterable, List, Generator
from argparse import ArgumentParser

def list_files_in_directory(input_dir: str) -> List[str]:
    """List all files in the given directory."""
    return [os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith('.mrc')]

def chunked(iterable: Iterable, size: int) -> Generator[List, None, None]:
    """Yield successive `size`-sized lists from `iterable`."""
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(size):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk

def read_filelist(filelist_path):
    """
    Read a job list file and return a list of file paths.

    Args:
        filelist_path (str): Path to the job list file.

    Returns:
        list: List of file paths.
    """
    with open(filelist_path, 'r') as f:
        file_paths = [line.strip() for line in f if line.strip()]
    return file_paths

def create_job_list(data_dir_path, job_file_path,seg_model_path, save_dir_path,
                    nb_of_jobs, batch_size):
    """
    Create a job list file containing paths of all files in the input directory.

    Args:
        data_dir_path (str): Path to the directory containing MRC files.
        job_file_path (str): Path to the job file to create (will be overwritten).
        seg_model_path (str): Path to the segmentation model directory or file.
        nb_of_jobs (int): Maximum number of job batches to write (use None for all).
        batch_size (int): Number of files per batch written on each line.
    """

    filelist = list_files_in_directory(data_dir_path)
    print(f"Found {len(filelist)} MRC files in {data_dir_path}.")
    prefix = ("module load miniconda; conda activate ves_seg; "
              "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH; "
              f"export SEGMENTATION_DIR={seg_model_path}; "
              f"export SAVEDIR={save_dir_path}; "
              f"export INPUTDIR={data_dir_path}; ")
    if nb_of_jobs is None:
        nb_of_jobs = len(filelist)
    nb_of_jobs_iter = 0

    with open(job_file_path, 'w') as f:
        for batch in chunked(filelist, batch_size):
            # process the whole batch (e.g., submit jobs in groups)
            f.write(prefix)
            for file_path in batch:
                file_path = endwith_mrc(strip_leading_dot_slash(file_path))
                f.write(f"bash seg_subtract_one_file.sh {file_path}  ; ")
            f.write("\n")  # Separate batches with a newline
            nb_of_jobs_iter += 1
            if nb_of_jobs_iter > nb_of_jobs:
                print(f"Reached the maximum number of jobs: {nb_of_jobs}")
                break

def strip_leading_dot_slash(s: str) -> str:
    """Remove leading `./` if present."""
    return s[2:] if s.startswith("./") else s

def endwith_mrc(s: str) -> bool:
    """Check if the string ends with `.mrc`."""
    return s[:-4] if s.lower().endswith(".mrc") else s

if __name__ == "__main__":
    args = ArgumentParser(description="Create job list for processing MRC files")
    args.add_argument("-ddp","--data_dir_path", type=str, help="Path to the directory containing MRC files")
    args.add_argument("-jfp", "--job_file_path", type=str, default=None, help="Path to the job file to create")
    args.add_argument("-segmp", "--seg_model_path", type=str, help="Path to the segmentation model")
    args.add_argument("-savedp", "--save_dir_path", type=str, help="Path to the dir where results will be saved")
    parsed_args = args.parse_args()
    create_job_list(**vars(parsed_args), nb_of_jobs=None,batch_size=8)
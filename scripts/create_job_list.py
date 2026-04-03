import os
import torch
from typing import Iterable, List, Generator, Optional
from argparse import ArgumentParser

def list_files_in_directory(input_dir: str) -> List[str]:
    """
    List all files in the given directory that have a `.mrc` extension.

    Args:
        input_dir (str): Path to the directory to scan for files.

    Returns:
        List[str]: A list of `.mrc` file names in the directory.
    """
    return [ os.path.join(input_dir,f) for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith('.mrc')]

def chunked(iterable: Iterable, size: int) -> Generator[List, None, None]:
    """
    Yield successive chunks of a specified size from an iterable.

    Args:
        iterable (Iterable): The input iterable to split into chunks.
        size (int): The size of each chunk.

    Yields:
        Generator[List, None, None]: A generator yielding lists of items from the iterable.
    """
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
        List[str]: List of file paths.
    """
    with open(filelist_path, 'r') as f:
        file_paths = [line.strip() for line in f if line.strip()]
    return file_paths

def create_job_list(data_dir_path, job_file_path,save_dir_path,
                    nb_of_jobs, batch_size, file_mode, save_angle_flag=0, save_sub_flag=0):
    """
    Create a job list file containing paths of all files in the input directory.

    Args:
        data_dir_path (str): Path to the directory containing MRC files.
        job_file_path (str): Path to the job file to create (will be overwritten).
        nb_of_jobs (int): Maximum number of job batches to write (use None for all).
        batch_size (int): Number of files per batch written on each line.
        file_mode (str): Mode to write into a jobfile, "w" or "a".
    Returns:
        None
    """

    filelist = list_files_in_directory(data_dir_path)
    filelist = delete_processed_files_from_fnamelist(filelist, save_dir_path, save_angle_flag, save_sub_flag)

    if len(filelist) == 0:
        print(f"No unprocessed MRC files found in {data_dir_path}. Exiting.")
        return
    else:
        print(f"Found {len(filelist)} unprocessed MRC files in {data_dir_path}")
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    prefix = ("module reset; module load miniconda; conda activate ves_seg;"
        # f"export LD_LIBRARY_PATH={torch_lib_path}:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH; "
        # f"export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH:{torch_lib_path};"
        # f"export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH;"
        # f"export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH;"
        # f"export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH;"
        #f"export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH;" 
        f"export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH;"
        f"for d in $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/lib; do [ -d \"$d\" ] && export LD_LIBRARY_PATH=\"$d:$LD_LIBRARY_PATH\"; done;"
        f"export SAVEDIR={save_dir_path};"
        f"export SAVE_ANGLE={save_angle_flag};"
        f"export SAVE_SUB={save_sub_flag};")
    if nb_of_jobs is None:
        nb_of_jobs = len(filelist)
    nb_of_jobs_iter = 0

    with open(job_file_path, file_mode) as f:
        for batch in chunked(filelist, batch_size):
            # process the whole batch (e.g., submit jobs in groups)
            f.write(prefix)
            for filename in batch:
                filename= strip_leading_dot_slash(filename)
                f.write(f"bash scripts/seg_subtract_v1.sh {filename};")
            f.write("\n")  # Separate batches with a newline
            nb_of_jobs_iter += 1
            if nb_of_jobs_iter > nb_of_jobs:
                print(f"Reached the maximum number of jobs: {nb_of_jobs}")
                break

def delete_processed_files_from_fnamelist(fpaths, save_dir_path, save_angle_flag, save_sub_flag):
    """
    Filter out files that have already been processed.

    Args:
        fpaths (List[str]): List of file paths to check.
        save_dir_path (str): Path to the directory where processed files are saved.

    Returns:
        List[str]: List of unprocessed file names.
    """
    labels_dir = os.path.join(save_dir_path,"misc","angles")
    sub_dir = os.path.join(save_dir_path,"subtracted_mrc")
    unprocessed_fnames = []
    for fpath in fpaths:
        basename = os.path.splitext(os.path.basename(fpath))[0]
        file_was_processed = False
        if save_angle_flag==1 and save_sub_flag==1:
            if exists_ospath(os.path.join(labels_dir, basename + "_angles.mat")) and exists_ospath(os.path.join(sub_dir, basename + ".mrc")):
                file_was_processed = True
        elif save_angle_flag==1:
            if exists_ospath(os.path.join(labels_dir, basename + "_angles.mat")):
                file_was_processed = True
        elif save_sub_flag==1:
            if exists_ospath(os.path.join(sub_dir, basename + ".mrc")):
                file_was_processed = True

        if not file_was_processed:
            unprocessed_fnames.append(fpath)
        else:
            print(f"File already exist, skipping: {basename}")
    return unprocessed_fnames


def list_nonempty_mrc_subdirs(root: str) -> List[str]:
    """
    Return a list of immediate subdirectory paths under `root` that contain at least one .mrc file.
    Uses os.walk per subdirectory and stops walking that subdirectory once a match is found.
    """
    result: List[str] = []
    with os.scandir(root) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            for dirpath, _, filenames in os.walk(entry.path):
                if any(fn.lower().endswith('.mrc') for fn in filenames):
                    result.append(os.path.abspath(entry.path))
                    break
    return result

def strip_leading_dot_slash(s: str) -> str:
    """Remove leading `./` if present."""
    return s[2:] if s.startswith("./") else s

def delete_mrc_ext(s: str) -> bool:
    """Check if the string ends with `.mrc`."""
    return s[:-4] if s.lower().endswith(".mrc") else s

def exists_ospath(path: str) -> bool:
    """Return True if `path` exists (file or directory) using os.path."""
    return os.path.exists(path)

if __name__ == "__main__":
    args = ArgumentParser(description="Create job list for processing MRC files")
    args.add_argument("-ddp","--data_dir_path", type=str, help="Path to the directory containing MRC files")
    args.add_argument("-jfp", "--job_file_path", type=str, default=None, help="Path to the txt job file to create")
    args.add_argument("-savedp", "--save_dir_path", type=str, help="Path to the dir where results will be saved")
    args.add_argument("-n", "--nb_of_jobs", type=int, help="Number of jobs to create, if None all files will be processed", default=None)
    args.add_argument("--save_angle_flag", type=int, help="Flag to save angle information (1 to save, 0 otherwise)", default=0)
    args.add_argument("--save_sub_flag", type=int, help="Flag to save subtracted images (1 to save, 0 otherwise)", default=0)
    parsed_args = args.parse_args()
    sub_dirs = list_nonempty_mrc_subdirs(parsed_args.data_dir_path)
    batch_size = 10
    if len(sub_dirs) > 0:
        for sub_dir in sub_dirs:
            sub_dir_name = os.path.basename(os.path.normpath(sub_dir))
            job_file_path = os.path.join(parsed_args.job_file_path, f"job_list_{sub_dir_name}.txt") if parsed_args.job_file_path else f"job_list_{sub_dir_name}.txt"
            create_job_list(data_dir_path=sub_dir,
                            job_file_path=parsed_args.job_file_path,
                            save_dir_path=os.path.join(parsed_args.save_dir_path, sub_dir_name),
                            nb_of_jobs=args.nb_of_jobs,
                            batch_size=batch_size,
                            file_mode="a",
                            save_angle_flag=parsed_args.save_angle_flag,
                            save_sub_flag=parsed_args.save_sub_flag)
    else:
        create_job_list(**vars(parsed_args),batch_size=batch_size, file_mode="w")
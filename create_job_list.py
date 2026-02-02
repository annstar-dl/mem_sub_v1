import os
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
    return [ f for f in os.listdir(input_dir)
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

def create_job_list(data_dir_path, job_file_path,seg_model_path, save_dir_path,
                    nb_of_jobs, batch_size, file_mode):
    """
    Create a job list file containing paths of all files in the input directory.

    Args:
        data_dir_path (str): Path to the directory containing MRC files.
        job_file_path (str): Path to the job file to create (will be overwritten).
        seg_model_path (str): Path to the segmentation model directory or file.
        nb_of_jobs (int): Maximum number of job batches to write (use None for all).
        batch_size (int): Number of files per batch written on each line.
        file_mode (str): Mode to write into a jobfile, "w" or "a".
    Returns:
        None
    """

    filelist = list_files_in_directory(data_dir_path)
    filelist = delete_processed_files_from_fnamelist(filelist, save_dir_path)
    print(f"Found {len(filelist)} MRC files in {data_dir_path}.")
    prefix = ("module load miniconda; conda activate ves_seg; "
              "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH; "
              f"export SEGMENTATION_DIR={seg_model_path}; "
              f"export SAVEDIR={save_dir_path}; "
              f"export INPUTDIR={data_dir_path}; ")
    if nb_of_jobs is None:
        nb_of_jobs = len(filelist)
    nb_of_jobs_iter = 0

    with open(job_file_path, file_mode) as f:
        for batch in chunked(filelist, batch_size):
            # process the whole batch (e.g., submit jobs in groups)
            f.write(prefix)
            for filename in batch:
                filename= delete_mrc_ext(strip_leading_dot_slash(filename))
                f.write(f"bash seg_subtract_one_file.sh {filename};")
            f.write("\n")  # Separate batches with a newline
            nb_of_jobs_iter += 1
            if nb_of_jobs_iter > nb_of_jobs:
                print(f"Reached the maximum number of jobs: {nb_of_jobs}")
                break

def delete_processed_files_from_fnamelist(fnames, save_dir_path):
    """
    Filter out files that have already been processed.

    Args:
        fnames (List[str]): List of file names to check.
        save_dir_path (str): Path to the directory where processed files are saved.

    Returns:
        List[str]: List of unprocessed file names.
    """
    mrc_reconstruction_dir = os.path.join(save_dir_path, "reconstructions", "subtracted_mrc")
    unprocessed_fnames = []
    for fname in fnames:
        if not exists_ospath(os.path.join(mrc_reconstruction_dir, fname)):
            unprocessed_fnames.append(fname)
        else:
            print(f"File already exist, skipping: {fname}")
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
    args.add_argument("-segmp", "--seg_model_path", type=str, help="Path to the segmentation model")
    args.add_argument("-savedp", "--save_dir_path", type=str, help="Path to the dir where results will be saved")
    args.add_argument("-n", "--nb_of_jobs", type=int, help="Number of jobs to create, if None all files will be processed", default=None)
    parsed_args = args.parse_args()
    sub_dirs = list_nonempty_mrc_subdirs(parsed_args.data_dir_path)
    if len(sub_dirs) > 0:
        for sub_dir in sub_dirs:
            sub_dir_name = os.path.basename(os.path.normpath(sub_dir))
            job_file_path = os.path.join(parsed_args.job_file_path, f"job_list_{sub_dir_name}.txt") if parsed_args.job_file_path else f"job_list_{sub_dir_name}.txt"
            create_job_list(data_dir_path=sub_dir,
                            job_file_path=parsed_args.job_file_path,
                            seg_model_path=parsed_args.seg_model_path,
                            save_dir_path=os.path.join(parsed_args.save_dir_path, sub_dir_name),
                            nb_of_jobs=args.nb_of_jobs,
                            batch_size=8,
                            file_mode="a")
    else:
        create_job_list(**vars(parsed_args),batch_size=8, file_mode="w")
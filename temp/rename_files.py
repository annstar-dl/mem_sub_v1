import os
import shutil

def rename_files(src_dir, files_dir, output_dir):
    """
    Rename files in the given list from old_ext to new_ext.

    :param file_list: List of file names (strings)
    :param old_ext: Old file extension (string)
    :param new_ext: New file extension (string)
    :return: List of renamed file names
    """
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        prefix = fname.split("_")[0]
        rest_of_name = "_".join(fname.split("_")[1:])
        rest_of_name, ext = os.path.splitext(rest_of_name)
        old_fpath = os.path.join(files_dir, f"{rest_of_name}.mrc")
        new_fpath = os.path.join(output_dir, f"{prefix}_{rest_of_name}.mrc")
        shutil.copy(old_fpath,new_fpath)

if __name__ == "__main__":
    src_dir = r"/home/astar/Projects/vesicles_data/iclr_experiments/experiments/VSM/images"
    files_dir = r"/home/astar/Projects/vesicles_data/iclr_experiments/experiments/VSM/SA_algorithm/subtracted_mrc"
    output_dir = r"/home/astar/Projects/vesicles_data/iclr_experiments/experiments/VSM/SA_algorithm/subtracted_mrc_renamed"
    rename_files(src_dir, files_dir, output_dir)
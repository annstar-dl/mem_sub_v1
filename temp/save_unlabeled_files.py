import os
import shutil


def save_unlabeled_files(input_dir: str, output_dir: str, labeled_files: set):
    """
    Save unlabeled files from the input directory to the output directory.

    Args:
        input_dir (str): Path to the directory containing input files.
        output_dir (str): Path to the directory where unlabeled files will be saved.
        labeled_files (set): Set of filenames that are labeled.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        filename_no_ext, _ = os.path.splitext(filename)
        if filename_no_ext not in labeled_files:
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(output_dir, filename)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"Saved unlabeled file: {filename}")

def find_labeled_files(label_dir: str, setname) -> set:
    """
    Find all labeled files in the given directory.

    Args:
        label_dir (str): Path to the directory containing labeled files.
    Returns:
        set: Set of filenames that are labeled.
    """
    labeled_files = set()
    for filename in os.listdir(label_dir):
        if setname in filename:
            labeled_files.add(os.path.splitext(filename)[0])
    return labeled_files

if __name__ == "__main__":
    input_directory = "/synology/membranes/data/20250226_H1-sonic/20250226_H1-sonic_jpg/part1/images_jpeg/part1_ds"
    label_directory = "/home/astar/Projects/membrane_detection/data/unseparated_membrane/images"
    output_directory = "/synology/membranes/data/20250226_H1-sonic/20250226_H1-sonic_jpg_unlabeled/part1"
    setname = "H1-sonic"
    labeled_files_set = find_labeled_files(label_directory, setname)
    save_unlabeled_files(input_directory, output_directory, labeled_files_set)
import shutil
import os

def extract_uprocessed_files(input_dir, target_dir, output_dir):
    """
    Extract unprocessed files from the input directory and save them to the output directory.

    Args:
        input_dir (str): Path to the directory containing the files to extract.
        output_dir (str): Path to the directory where the extracted files will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_names =  [os.path.splitext(file_name)[0] for file_name in os.listdir(input_dir)]
    target_names = [os.path.splitext(file_name)[0][:-7] for file_name in os.listdir(target_dir)]
    input_names = set(input_names)
    target_names = set(target_names)
    unprocessed_files = list(input_names.difference(target_names))
    unprocessed_files = [os.path.join(input_dir,file_name) + ".mrc" for file_name in unprocessed_files]

    for file in unprocessed_files:
        shutil.copy(file, output_dir)
        print(f"Copied {file} to {output_dir}")

if __name__ == "__main__":
    maindir = r"/vast/palmer/scratch/sigworth/as4873"
    input_dir = os.path.join(maindir, "Prestin_20250710/Micrographs")
    target_dir = os.path.join(maindir, "Prestin_angles/misc/angles")
    output_dir = os.path.join(maindir, "Prestin_uprocessed")
    extract_uprocessed_files(input_dir, target_dir, output_dir)
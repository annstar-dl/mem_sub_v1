import os

def compare_directories(dir1: str, dir2: str) -> None:
    """
    Compare files in two directories and print files that are in dir1 but not in dir2.

    Args:
        dir1 (str): Path to the first directory.
        dir2 (str): Path to the second directory.
    """
    files_dir1 = set(os.listdir(dir1))
    files_dir2 = set(os.listdir(dir2))

    common = files_dir1 & files_dir2
    combined = files_dir1 | files_dir2  # use union, not +

    if common:
        print("Some files are present in both directories:")
        for filename in sorted(common):
            print(filename)
    else:
        print(f"All files are unique between the two directories.")

    print(f"Number of combined files: {len(combined)}")


if __name__ == "__main__":
    directory1 = r"/media/astar/PortableSSD/for_labeling/separated_TG_ST/ST/prestin_20250710"
    directory2 = r"/media/astar/PortableSSD/for_labeling/separated_TG_ST/TG/prestin_20250710"
    compare_directories(directory1, directory2)
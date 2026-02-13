import shutil
import os


def compare_files(old_dir, new_dir, output_dir):

    old_names =  [os.path.splitext(file_name)[0] for file_name in os.listdir(old_dir)]
    old_names = set(old_names)
    new_names_1 = [os.path.splitext(file_name)[0] for file_name in os.listdir(os.path.join(new_dir,"images","val"))]
    new_names_1 = set(new_names_1)
    diff_val = list(new_names_1.difference(old_names))
    print(f"{len(diff_val)} Validation files not in train set: {diff_val}")
    new_names_2 = [os.path.splitext(file_name)[0] for file_name in os.listdir(os.path.join(new_dir, "images", "test"))]
    new_names_2 = set(new_names_2)
    diff_test = list(new_names_2.difference(old_names))
    print(f"{len(diff_test)} test files not in train set: {diff_test}")
    for filename in diff_val:
        shutil.copy(os.path.join(new_dir,"images","val", filename+".jpg"), os.path.join(output_dir,"images"))

        shutil.copy(os.path.join(new_dir, "labels", "val",filename+".png"), os.path.join(output_dir,"labels"))
    for filename in diff_test:
        shutil.copy(os.path.join(new_dir,"images","test", filename+".jpg"), os.path.join(output_dir,"images"))
        shutil.copy(os.path.join(new_dir, "labels", "test",filename + ".png"), os.path.join(output_dir, "labels"))


if __name__ == "__main__":
    old_dir = r"/home/astar/Projects/membrane_detection/data/membrane/images/train"
    new_dir = r"/home/astar/Projects/membrane_detection/data/2025_sep+dec_separated_membrane"
    output_dir = r"/home/astar/Projects/membrane_detection/temp/not_in_both_train_sets"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir,"images")):
        os.makedirs(os.path.join(output_dir,"images"))
    if not os.path.exists(os.path.join(output_dir,"labels")):
        os.makedirs(os.path.join(output_dir,"labels"))
    compare_files(old_dir, new_dir, output_dir)
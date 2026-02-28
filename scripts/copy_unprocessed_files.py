from create_job_list import list_files_in_directory, delete_processed_files_from_fnamelist
import os
import shutil
import argparse

def main(data_dir_path, save_dir_path, new_save_path, save_angle_flag=0, save_sub_flag=0):
    filelist = list_files_in_directory(data_dir_path)
    filelist = delete_processed_files_from_fnamelist(filelist, save_dir_path, save_angle_flag, save_sub_flag)
    for file in filelist:
        src_path = os.path.join(data_dir_path, file)
        dst_path = os.path.join(new_save_path, file)
        shutil.copy(src_path, dst_path)
        print(f"Copied {src_path} to {dst_path}")

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Copy unprocessed MRC files to a new directory.")
    args.add_argument("-ddp","--data_dir_path", type=str, help="Path to the directory containing MRC files")
    args.add_argument("-sdp","--save_dir_path", type=str, help="Path to the directory where processed files are saved")
    args.add_argument("-nsp","--new_save_path", type=str, help="Path to the directory where unprocessed files will be copied")
    args.add_argument("-saf","--save_angle_flag", type=int, default=1, help="Flag indicating whether angle files are saved (1) or not (0)")
    args.add_argument("-ssf","--save_sub_flag", type=int, default=1, help="Flag indicating whether subtracted MRC files are saved (1) or not (0)")
    args = args.parse_args()
    main(args.data_dir_path, args.save_dir_path, args.new_save_path, args.save_angle_flag, args.save_sub_flag)
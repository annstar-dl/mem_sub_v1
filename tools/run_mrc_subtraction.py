import os
import numpy as np
import time
from mem_sub.membrane_est.utils import save_im
from mem_sub.membrane_est.membrane_estimation import membrane_estimation
from tqdm import tqdm
import argparse
from scipy.io import savemat
from mem_sub.mrc_tools.mrc_utils import load_mrc, downsample_micrograph, save_im_mrc_same_size, \
    upsample_micrograph, FILE_TYPES
from mem_sub.membrane_est.utils import read_parameters_from_yaml_file, read_img
import pandas as pd


def read_mrc(fpath):
    #check if the file exists
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"File {fpath} does not exist.")
    img, header, voxel_size = load_mrc(fpath)
    if np.any(np.isnan(img)):
        raise ValueError(f"Damaged mrc file, Nan in {os.path.basename(fpath)}")
    img = img.astype(np.float64)
    return img, header, voxel_size


def process_file(file_name,args: argparse.Namespace):
    """Process a single MRC file to subtract membranes and save the results."""
    basename = os.path.splitext(file_name)[0] #file name without extension
    # read radius size from parameters file
    parameters = read_parameters_from_yaml_file()
    border = parameters["r"]  # Border size for fuzzy mask
    #read micrograph from mrc file
    img, header, voxel_size = read_mrc(os.path.join(args.imgs_path, file_name))
    if voxel_size[0]>4.5:
        raise ValueError(f"Voxel size {voxel_size[0]} is larger than 4.0 Angstrom. "
                         f"Your micrograph is probably downsampled, use micrograph of original size.")
    #read downsampled membrane mask of a micrograph from png file
    print(f"Mask path, {os.path.join(args.masks_path, basename + ".png")}")
    mask = read_img(os.path.join(args.masks_path, basename + ".png"), True)
    # downsample the micrograph if needed
    img_ds = downsample_micrograph(img, voxel_size[0], border, "center")
    # check if the image is the same size as the mask
    if img_ds.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image {file_name} and mask {basename}.png must have the same dimensions. "
            f"Image shape: {img.shape}, Mask shape: {mask.shape}")

    # run membrane subtraction algorithm
    st_time = time.time()
    membrane_ds, angle_dict = membrane_estimation(img_ds, mask, border if np.any(img.shape > img_ds.shape) else 0)
    print(f"Membrane estimation for {file_name} took {time.time() - st_time:.2f} seconds.")

    if args.save_subtraction:
        # upsample the membrane estimate to the original size
        membrane = upsample_micrograph(membrane_ds, img.shape, voxel_size[0], "center")
        # subtract membrane from the original image
        sub_img = img - membrane
        #save subtracted image
        for fmt in args.out_format_sub:
            if "mat" in args.out_format_sub:
                # Save as .mat file if specified
                savemat(os.path.join(args.subtracted_mat_path, basename + ".mat"),
                        {'img': img, 'label': mask, 'mem': membrane, 'sub': sub_img})
            if "mrc" in args.out_format_sub:
                # Save as .png file if specified
                save_im_mrc_same_size(sub_img, os.path.join(args.subtracted_mrc_path, basename + ".mrc"), header)
            if not fmt in ["mat","mrc"]:
                # Save as .png file if specified
                sub_img_ds = downsample_micrograph(sub_img, voxel_size[0], border, "center")
                save_im(sub_img_ds, os.path.join(args.subtracted_ds_path, basename + "." + fmt))

        if len(args.out_format_mem)!=0:
            for fmt in args.out_format_mem:
                if fmt == "mrc":
                    save_im_mrc_same_size(membrane, os.path.join(args.membrane_path, basename + ".mrc"), header)
                if fmt == "npy":
                    np.save(os.path.join(args.membrane_path, basename + ".npy"), membrane)
                if not fmt in ["mrc","npy"]:
                    save_im(membrane_ds, os.path.join(args.membrane_path+"_ds", basename + "." + fmt))

    if args.save_angle:
        save_angle_info(args, basename, angle_dict)

def save_angle_info(args, basename, angle_dict):
    """Save angle information to a .mat file.
    Args:
        args (argparse.Namespace): Command-line arguments.
        basename (str): Base name of the file.
        angle_dict (dict): Dictionary containing angle information.
        """
    print("Saving angle information...")
    df = pd.DataFrame(angle_dict)
    df.to_csv(os.path.join(args.angle_path, basename + "_angles.csv"), index=False)
    savemat(os.path.join(args.angle_path, basename + "_angles.mat"), angle_dict)

def process_dir(args):
    """Process all MRC files in a directory to subtract membranes and save the results."""
    for file_name in tqdm(os.listdir(args.imgs_path), desc="Processing images"):
        if not file_name.endswith("mrc"):
            continue
        try:
            process_file(file_name,args)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

def main(args):
    """Main function to set up directories and process files."""
    for fmt in args.out_format_mem:
        if fmt == "mrc" or fmt=="npy":
            args.membrane_path = os.path.join(args.output_path, "misc", "membranes")
            if not os.path.exists(args.membrane_path):
                os.makedirs(args.membrane_path)
        else:
            args.membrane_path_ds = os.path.join(args.output_path, "misc", "membranes_ds")
            if not os.path.exists(args.membrane_path_ds):
                os.makedirs(args.membrane_path_ds)

    for fmt in args.out_format_sub:
        if fmt=="mrc":
            subtracted_path = os.path.join(args.output_path, "subtracted_mrc")
            args.subtracted_mrc_path = subtracted_path
            if not os.path.exists(subtracted_path):
                os.makedirs(subtracted_path)
        elif fmt=="mat":
            subtracted_path = os.path.join(args.output_path,"misc",f"subtracted_{fmt}")
            args.subtracted_mat_path = subtracted_path
            if not os.path.exists(subtracted_path):
                os.makedirs(subtracted_path)
        else:
            subtracted_path = os.path.join(args.output_path,"misc",f"subtracted_{fmt}_ds")
            args.subtracted_ds_path = subtracted_path
        if not os.path.exists(subtracted_path):
            os.makedirs(subtracted_path)


    args.masks_path = os.path.join(args.output_path,"misc", "labels")
    if args.save_angle:
        args.angle_path = os.path.join(args.output_path,"misc", "angles")
        if not os.path.exists(args.angle_path):
            os.makedirs(args.angle_path)
    #process images
    if os.path.isdir(args.imgs_path):
        process_dir(args)
    else:
        if not args.imgs_path.lower().endswith(tuple(FILE_TYPES)):
            raise ValueError(f"File {args.imgs_path} is not an MRC file.")

        file_name = os.path.basename(args.imgs_path)
        args.imgs_path = os.path.dirname(args.imgs_path)
        process_file(file_name,args)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Process membrane images and subtract membranes.")
    parser.add_argument("-dp","--output_path", type=str,help="Directory path containing folders with images and labels")
    parser.add_argument("-ip","--imgs_path", type=str,  help="Directory of file path containing mrc micrograph/micrographs")
    parser.add_argument("--out_format_sub", nargs="*", default=["png"], help="List of file format to save subracted images. Choices are mat, png, mrc formats")
    parser.add_argument("--out_format_mem", nargs="*", default=["npy"],  help="List of file format to save estimates of membrane images. Choices are mrc, png, npy formats")
    parser.add_argument("-ang","--save_angle", action="store_true", help="Whether to save angle for every grid point information or not.")
    parser.add_argument("-sub", "--save_subtraction", action="store_true", help="Whether to do membrane subtraction or not.")
    args = parser.parse_args()
    main(args)




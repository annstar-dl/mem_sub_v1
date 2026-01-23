import os
import numpy as np
from PIL import Image
from membrane_estimation import membrane_estimation
from tqdm import tqdm
import argparse
from scipy.io import savemat
from mrc_utils import load_mrc, downsample_micrograph, save_im_mrc_same_size, \
    upsample_micrograph, new_shape_mrc_downsampling, FILE_TYPES
from sub_utils import read_dict_from_yaml_file
from bg_estimation import get_background
import time

def read_img(fpath, mask=False):
    #check if the file exists
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"File {fpath} does not exist.")
    else:
        img = Image.open(fpath)
        img = np.array(img,dtype = np.float64)
        if mask:
            img = img - np.min(img)
            img = img / np.max(img)
            img = ( img > 0.5 ).astype(np.float64)
    return img

def read_mrc(fpath):
    #check if the file exists
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"File {fpath} does not exist.")
    img, header, voxel_size = load_mrc(fpath)
    img = img.astype(np.float64)
    return img, header, voxel_size

def save_im(img, fpath):
    """Save the image to a file after normalizing it to the range [0, 255].
    Args:
        img (numpy.ndarray): Image array to save.
        fpath (str): Path to save the image file.
    """
    img = img - np.min(img)
    img = img / np.max(img) * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img,"L")
    img.save(fpath)


def process_file(args: argparse.Namespace):
    """Process a single MRC file to subtract membranes and save the results."""
    basename = os.path.splitext(args.file_name)[0] #file name without extension
    # read radius size from parameters file
    parameters = read_dict_from_yaml_file()
    border = parameters["r"]  # Radius of neighboring around grid point
    #read micrograph from mrc file
    img, header, voxel_size = read_mrc(os.path.join(args.imgs_path, args.file_name))
    #read downsampled membrane mask of a micrograph from png file
    mask = read_img(os.path.join(args.masks_path, basename + ".png"), True)
    if np.any(np.isnan(img)):
        raise ValueError(f"Empty file, Nan in {args.file_name}")

    img_ds = downsample_micrograph(img, voxel_size[0], border, "center")
    # check if the image is the same size as the mask
    if img_ds.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image {args.file_name} and mask {basename}.png must have the same dimensions. "
            f"Image shape: {img.shape}, Mask shape: {mask.shape}")

    # run membrane subtraction algorithm
    imgout_ds = membrane_estimation(img_ds, mask, border if np.any(img.shape > img_ds.shape) else 0)
    # upsample the membrane estimate to the original size
    imgout = upsample_micrograph(imgout_ds, img.shape, voxel_size[0], border, "center")
    sub_img = img - imgout
    # add background back to the subtracted image
    if "mat" in args.out_format:
        # Save as .mat file if specified
        savemat(os.path.join(args.imgsout_path + "_mat", basename + ".mat"),
                {'img': img, 'label': mask, 'mem': imgout, 'sub': sub_img})
    if "jpeg" in args.out_format:
        # Save as .png file if specified
        save_im(sub_img, os.path.join(args.imgsout_path + "_jpeg", basename + ".jpeg"))
    if "png" in args.out_format:
        # Save as .png file if specified
        save_im(sub_img, os.path.join(args.imgsout_path + "_png", basename + ".png"))
    if "mrc" in args.out_format:
        # Save as .png file if specified
        save_im_mrc_same_size(sub_img, os.path.join(args.imgsout_path + "_mrc", basename + ".mrc"), header)
        if args.save_reconstruction:
            save_im_mrc_same_size(imgout, os.path.join(args.imgsout_reconstructed_path, basename + ".mrc"), header)
    if args.save_reconstruction:
        save_im(imgout, os.path.join(args.imgsout_reconstructed_path, basename + ".tif"))
        np.save(os.path.join(args.imgsout_reconstructed_path, basename + ".npy"), imgout)

def process_dir(args):
    """Process all MRC files in a directory to subtract membranes and save the results."""
    for file_name in tqdm(os.listdir(args.imgs_path), desc="Processing images"):
        if not file_name.endswith("mrc"):
            continue
        args.file_name = file_name
        try:
            process_file(args)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

def main(args):
    """Main function to set up directories and process files."""
    imgout_mainpath = os.path.join(args.output_path, "reconstructions")
    if args.save_reconstruction:
        args.imgsout_reconstructed_path = os.path.join(imgout_mainpath, "reconstructed_membranes")
        if not os.path.exists(args.imgsout_reconstructed_path):
            os.makedirs(args.imgsout_reconstructed_path)
    args.imgsout_path = os.path.join(imgout_mainpath,"subtracted")
    for fmt in args.out_format:
        imgsout_path = args.imgsout_path + f"_{fmt}"
        if not os.path.exists(imgsout_path):
            os.makedirs(imgsout_path)
    args.masks_path = os.path.join(args.output_path, "labels")
    if args.file_name is None:
        process_dir(args)
    else:
        if not args.file_name.lower().endswith(tuple(FILE_TYPES)):
            raise ValueError(f"File {args.file_name} is not an MRC file.")
        process_file(args)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Process membrane images and subtract membranes.")
    parser.add_argument("-dp","--output_path", type=str,help="Directory path containing folders with images and labels")
    parser.add_argument("-ip","--imgs_path", type=str,  help="Directory path containing mrc micrographs")
    parser.add_argument("-s","--sigma", type=float, default=24.0, help="Sigma for Gaussian filter to flatten background.")
    parser.add_argument("--out_format", nargs="+", default=["png"], help="List of file format to save subracted images. Choices are .mat,.png.,mrc formats")
    parser.add_argument("--save_reconstruction", action="store_true", default=False, help="Save the reconstructed membranes.")
    parser.add_argument("-fn","--file_name",type=str, default=None,help="Name of file to convert (default: None), if None process all files in the folder")
    args = parser.parse_args()
    main(args)




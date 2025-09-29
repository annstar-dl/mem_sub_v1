import os
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from sympy.tensor.tensor import substitute_indices

from membrane_subtract_mrc import membrane_subtract
from tqdm import tqdm
import argparse
from scipy.io import savemat
from mrc2jpg import load_mrc, downsample_mrc
import mrcfile
from run import read_img



def read_mrc(fpath):
    #check if the file exists
    print(f"Processing image: {fpath}")
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
    img = img.detach().cpu().numpy()
    img = img.astype(np.uint8)
    img = Image.fromarray(img,"L")
    img.save(fpath)

def save_im_mrc(img, fpath, header):
    """Save the image to a MRC file.
    Args:
        img (numpy.ndarray): Image array to save.
        fpath (str): Path to save the image file.
    """
    img = img.detach().cpu().numpy()
    with mrcfile.new(fpath, overwrite=True) as mrc_new:
        #Do not change nx,ny,nz values in the header
        #Do not chnage dmin, dmax, dmean, and rms
        field_names = mrc_new.header.dtype.names
        for field in field_names:
            if field!="nx" and field!="ny" and field!="nz" and field!="dmin" and field!="dmax" and field!="dmean" and field!="rms":
                mrc_new.header[field] = header[field]
        mrc_new.set_data(img.astype(np.float32))

def upsample_to_original(img_downsampled, original_shape):
    """Upsample the downsampled image to the original shape using nearest neighbor interpolation.
    Args:
        img_downsampled (numpy.ndarray): Downsampled image array.
        original_shape (tuple): Original shape of the image (height, width).
    Returns:
        numpy.ndarray: Upsampled image array.
    """
    img_downsampled = img_downsampled.detach().cpu().numpy()
    upsampled_img = np.array(Image.fromarray(img_downsampled).resize((original_shape[1], original_shape[0]), Image.NEAREST))
    return upsampled_img

def main(args):
    imgs_path = os.path.join(args.dataset_path, args.imgs_dir)
    masks_path = os.path.join(args.dataset_path, args.masks_dir)
    for fpath in tqdm(os.listdir(imgs_path), desc="Processing images"):
        if not fpath.endswith("mrc"):
            continue
        basename = os.path.splitext(fpath)[0]
        #scip if the output files already exist
        if os.path.exists(os.path.join(args.imgsout_path + "_jpeg", basename + ".jpeg")) and \
              os.path.exists(os.path.join(args.imgsout_path + "_mat", basename + ".mat")) and \
                os.path.exists(os.path.join(args.imgsout_path + "_mrc", basename + ".mrc")):
            print(f"Output for {basename} already exists. Skipping...")
            continue

        img_fname = basename+".mrc"
        img, header, voxel_size = read_mrc(os.path.join(imgs_path, img_fname))
        mask = read_img(os.path.join(masks_path, basename + ".png"),"png",1,True)
        img_downsampled = downsample_mrc(img, voxel_size)
        # check if the image is the same size as the mask
        if img.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image {img_fname} and mask {basename}.png must have the same dimensions. Image shape: {img.shape}, Mask shape: {mask.shape}")
        # run membrane subtraction algorithm
        imgout_downsampled = membrane_subtract(img_downsampled, mask)
        #upsample the membrane estimate to the original size
        imgout = upsample_to_original(img_downsampled, img.shape)
        sub_img = img - imgout
        # add background back to the subtracted image
        if "mat" in args.out_format:
            # Save as .mat file if specified
            savemat(os.path.join(args.imgsout_path+"_mat", basename + ".mat"),
                    {'img': img, 'label': mask, 'mem': imgout.numpy(), 'sub': sub_img.numpy()})
        if "jpeg" in args.out_format:
            # Save as .png file if specified
            save_im(sub_img, os.path.join(args.imgsout_path+"_jpeg", basename + ".jpeg"))
        if "png" in args.out_format:
            # Save as .png file if specified
            save_im(sub_img, os.path.join(args.imgsout_path+"_png", basename + ".png"))
        if "mrc" in args.out_format:
            # Save as .png file if specified
            save_im_mrc(sub_img, os.path.join(args.imgsout_path+"_mrc", basename + ".mrc"), header)
        if args.save_reconstruction:
            save_im(imgout, os.path.join(args.imgsout_reconstructed_path, basename + ".tif"))
            np.save(os.path.join(args.imgsout_reconstructed_path, basename + ".npy"), imgout)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Process membrane images and subtract membranes.")
    parser.add_argument("-dp","--dataset_path", type=str,help="Directory containing folders with images and labels")
    parser.add_argument("-id","--imgs_dir", type=str, default="images", help="Directory containing images.")
    parser.add_argument("-md","--masks_dir", type=str, default="labels", help="Directory containing labels for images.")
    parser.add_argument("-s","--sigma", type=float, default=24.0, help="Sigma for Gaussian filter to flatten background.")
    parser.add_argument("--in_format", type=str,choices=["mrc","jpeg","jpg","tif","tiff"],help="Format of input images. Default is mrc", default="mrc")
    parser.add_argument("--out_format", nargs="+", default="png", help="List of file format to save subracted images. Choices are .mat,.png.,mrc formats")
    parser.add_argument("-ds","--downsample_factor", type=int, default=1, help="Factor by which to downsample the images (default: 1).")
    parser.add_argument("--save_reconstruction", action="store_true", default=False, help="Save the reconstructed membranes.")
    args = parser.parse_args()

    main_path = args.dataset_path
    imgout_mainpath = os.path.join(main_path, "reconstructions")
    imgsout_subracted_path = os.path.join(imgout_mainpath,"subtracted")


    if args.save_reconstruction:
        imgsout_reconstructed_path = os.path.join(imgout_mainpath, "reconstructed_membranes")
        if not os.path.exists(imgsout_reconstructed_path):
            os.makedirs(imgsout_reconstructed_path)
        args.imgsout_reconstructed_path = imgsout_reconstructed_path
    args.imgsout_path = imgsout_subracted_path
    for fmt in args.out_format:
        imgsout_path = imgsout_subracted_path + f"_{fmt}"
        if not os.path.exists(imgsout_path):
            os.makedirs(imgsout_path)
    main(args)




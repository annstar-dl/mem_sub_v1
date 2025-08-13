import os
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from membrane_subtract import membrane_subtract
from tqdm import tqdm
import argparse
from scipy.io import savemat
from readmrc import load_mrc



def read_img(fpath,downsampling_factor=1,is_mrc=False):
    #check if the file exists
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"File {fpath} does not exist.")
    if is_mrc:
        img = load_mrc(fpath,downsample_factor=downsampling_factor).astype(np.float64)
    else:
        img = Image.open(fpath)
        img = np.array(img,dtype = np.float64)
    return img


def save_im(img, fpath):
    """Save the image to a file after normalizing it to the range [0, 255].
    Args:
        img (numpy.ndarray): Image array to save.
        fpath (str): Path to save the image file.
    """
    img = img.detach().cpu().numpy()
    img = img - np.min(img)
    img = img/np.max(img)
    img = (img*255).astype(np.uint8)
    img = Image.fromarray(img,"L")
    img.save(fpath)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Process membrane images and subtract membranes.")
    parser.add_argument("-dp","--dataset_path", type=str,help="Directory containing folders with images and labels")
    parser.add_argument("-id","--imgs_dir", type=str, default="images", help="Directory containing images.")
    parser.add_argument("-md","--masks_dir", type=str, default="labels", help="Directory containing labels for images.")
    parser.add_argument("-s","--sigma", type=float, default=24.0, help="Sigma for Gaussian filter to flatten background.")
    parser.add_argument("--mrc", action='store_true',help="Whether the input images are in .mrc format.")
    parser.add_argument("--save_as_mat", action='store_true',help="Whether to save images as .mat files instead of .png.")
    parser.add_argument("-ds","--downsample_factor", type=int, default=4, help="Factor by which to downsample the images (default: 4).")
    args = parser.parse_args()

    main_path = args.dataset_path
    imgout_mainpath = os.path.join(main_path, "reconstructions")
    imgsout_subracted_path = os.path.join(imgout_mainpath,"subtracted")
    imgsout_reconstructed_path = os.path.join(imgout_mainpath,"reconstructed_membranes")

    if not os.path.exists(imgsout_subracted_path):
        os.makedirs(imgsout_subracted_path)

    if not os.path.exists(imgsout_reconstructed_path):
        os.makedirs(imgsout_reconstructed_path)
    if args.save_as_mat:
        imgsout_path_mat = imgsout_subracted_path+ "_mat"
        if not os.path.exists(imgsout_path_mat):
            os.makedirs(imgsout_path_mat)

    imgs_path = os.path.join(main_path,args.imgs_dir)
    masks_path = os.path.join(main_path,args.masks_dir)
    for fpath in tqdm(os.listdir(imgs_path), desc="Processing images"):
        basename = os.path.splitext(fpath)[0]
        img_fname = basename + ".mrc" if args.mrc else basename + ".jpg"
        img = read_img(os.path.join(imgs_path,img_fname),args.downsample_factor,args.mrc)
        mask = read_img(os.path.join(masks_path, basename + ".png"))
        #check if the image is the same size as the mask
        if img.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Image {img_fname} and mask {basename}.png must have the same dimensions. Image shape: {img.shape}, Mask shape: {mask.shape}")
        imgout, sub_img = membrane_subtract(img,mask)
        #add background back to the subtracted image
        if args.save_as_mat:
            # Save as .mat file if specified
            savemat(os.path.join(imgsout_path_mat, basename + ".mat"), {'img': img, 'label': mask,'mem': imgout.numpy(),'sub': sub_img.numpy()})
        save_im(sub_img, os.path.join(imgsout_subracted_path,basename+".png"))
        save_im(imgout, os.path.join(imgsout_reconstructed_path,basename+".png"))



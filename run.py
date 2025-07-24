import os
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from membrane_subtract import membrane_subtract
from tqdm import tqdm
import argparse
from scipy.io import savemat

def read_membrane_img(fpath, flatten_bg=True, sigma=20.0):
    """Read membrane image from file and optionally flatten the background.
    Args:
        fpath (str): Path to the image file.
        flatten_bg (bool): Whether to flatten the background of the image.
        sigma (float): Standard deviation for Gaussian kernel if flattening is applied.
    Returns:
        numpy.ndarray: Image array, either with flattened background or original.
    """
    img = Image.open(fpath)
    img = np.array(img, dtype = np.float64)
    if flatten_bg:
        img = flatten_background(img, sigma)
    return img

def read_img(fpath):
    img = Image.open(fpath)
    img = np.array(img)
    return img

def flatten_background(img,sigma=20.0):
    """Flatten(even out) the background of the image using Gaussian smoothing.
    Args:
        img (numpy.ndarray): Input image array.
        sigma (float): Standard deviation for Gaussian kernel.
    Returns:
        numpy.ndarray: Image with flattened background.
        """
    img_smoothed = gaussian_filter(img, sigma)
    img = img - img_smoothed
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
    parser.add_argument("--imgs_path", type=str, default="images_fred_mck", help="Directory containing input images.")
    parser.add_argument("--masks_dir", type=str, default="labels_fred_mck", help="Directory containing labels for images.")
    parser.add_argument("--sigma", type=float, default=24.0, help="Sigma for Gaussian filter to flatten background.")
    parser.add_argument("--flatten_bg", action='store_true', help="Whether to flatten(even) the background of images.")
    parser.add_argument("--save_as_mat", action='store_true',help="Whether to save images as .mat files instead of .png.")
    args = parser.parse_args()

    main_path = os.path.dirname(args.imgs_path)
    dataset_name = os.path.basename(args.imgs_path)
    imgout_mainpath = os.path.join(main_path, dataset_name + "_reconstructions")
    imgout_mainpath += "" if not args.flatten_bg else "_flatten_bg"
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

    imgs_path = args.imgs_path
    masks_path = os.path.join(main_path,args.masks_dir)
    for fpath in tqdm(os.listdir(imgs_path), desc="Processing images"):
        fname = os.path.splitext(fpath)[0]
        img = read_membrane_img(os.path.join(imgs_path,fname+".jpg"), flatten_bg=args.flatten_bg, sigma=args.sigma)
        mask = read_img(os.path.join(masks_path,fname+".png"))
        imgout, sub_img = membrane_subtract(torch.tensor(img),torch.tensor(mask))
        if args.save_as_mat:
            # Save as .mat file if specified
            savemat(os.path.join(imgsout_path_mat, fname + ".mat"), {'img': img, 'sub': sub_img.numpy()})
        save_im(sub_img, os.path.join(imgsout_subracted_path,fname+".png"))
        save_im(imgout, os.path.join(imgsout_reconstructed_path,fname+".png"))



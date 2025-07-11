import os
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from membrane_subtract import membrane_subtract
from tqdm import tqdm
import argparse

def read_membrane_img(fpath, flatten_bg=True, sigma=20.0):
    """Read membrane image from file and optionally flatten the background.
    Args:
        fpath (str): Path to the image file.
        flatten_bg (bool): Whether to flatten the background of the image.
        sigma (float): Standard deviation for Gaussian kernel if flattening is applied.
    Returns:
        numpy.ndarray: Image array, either with flattened background or original.
    """
    img = Image.open(fpath).convert("L")
    img = np.array(img)
    if flatten_bg:
        img = flatten_background(img, sigma)
    return img

def read_img(fpath):
    img = Image.open(fpath).convert("L")
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
    parser.add_argument("--main_path", type=str, default="/home/astar/Projects/vesicles_data",
                        help="Main directory containing images and labels.")
    parser.add_argument("--imgsout_dir", type=str, default="imgout_fred_mck",help="Output directory for processed images.")
    parser.add_argument("--imgsout_subracted_dir", type=str, default="imgout_subracted_fred_mck",
                        help="Output directory for processed images.")
    parser.add_argument("--imgs_dir", type=str, default="images_fred_mck", help="Directory containing input images.")
    parser.add_argument("--masks_dir", type=str, default="labels_fred_mck", help="Directory containing labels for images.")
    parser.add_argument("--sigma", type=float, default=20.0, help="Sigma for Gaussian filter to flatten background.")
    parser.add_argument("--flatten_bg", action='store_true', help="Whether to flatten(even) the background of images.")
    args = parser.parse_args()

    imgsout_subracted_path = os.path.join(args.main_path,args.imgsout_subracted_dir)
    if not os.path.exists(imgsout_subracted_path):
        os.makedirs(imgsout_subracted_path)
    imgsout_path = os.path.join(args.main_path, args.imgsout_dir)
    if not os.path.exists(imgsout_path):
        os.makedirs(imgsout_path)
    imgs_path = os.path.join(args.main_path,args.imgs_dir)
    masks_path = os.path.join(args.main_path,args.masks_dir)
    for fpath in tqdm(os.listdir(imgs_path), desc="Processing images"):
        fname = os.path.splitext(fpath)[0]
        img = read_membrane_img(os.path.join(imgs_path,fname+".jpg"))
        mask = read_img(os.path.join(masks_path,fname+".png"))
        imgout, sub_img = membrane_subtract(torch.tensor(img),torch.tensor(mask))
        save_im(sub_img, os.path.join(imgsout_subracted_path,fname+".png"))
        save_im(imgout, os.path.join(imgsout_path,fname+".png"))



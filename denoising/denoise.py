import bm3d

from read_membrane_imgs import read_image, read_mask, get_background
import numpy as np
import copy
import os
from skimage import io
import argparse
from main_test_swinir import define_model
import torch
import importlib.util
import sys

file_path = r'/home/astar/Projects/SwinIR/main_test_swinir.py'
module_name = 'main_test_swinir'

spec = importlib.util.spec_from_file_location(module_name, file_path)
main_test_swinir = importlib.util.module_from_spec(spec)
sys.modules[module_name] = main_test_swinir
spec.loader.exec_module(main_test_swinir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ArgsSWINIR:
    def __init__(self, model_path):
        self.task = 'gray_dn'
        self.model_path = model_path

def swinir_denoising(img, model):
    """Denoise an image using SwinIR model."""
    # Convert image to 3-channel if it's grayscale
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    img = img # Normalize to [0, 1]
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img_tensor = torch.from_numpy(img).float()
    with torch.no_grad():
        denoised_tensor = model(img_tensor)
    denoised_img = denoised_tensor.squeeze().cpu().numpy()
    return denoised_img

def bm3d_denoising(img, sigma=0.1):
    """Denoise an image using BM3D algorithm."""
    # calculate the noise standard deviation from reagion outside the mask
    noisy_img = copy.deepcopy(img) # Normalize to [0, 1]
    #calculate the noise standard deviation from reagion outside the mask on the background flattened image
    #_,im_flatten = get_background(noisy_img,mask)
    #noise_region = im_flatten[mask < 0.5]
    sigma = sigma#np.std(noise_region)
    print(f"Estimated noise standard deviation: {sigma}")
    denoised_image = bm3d.bm3d(noisy_img, sigma_psd=sigma, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

    return denoised_image

def subract_membrane(img, denoised_image, mask):
    """Subtract membrane using the provided mask."""
    bg, _ = get_background(img,mask)
    img = img - (denoised_image-bg)*mask
    return img, bg

def save_image(fpath,img):
    """Save the image to a file after normalizing it to the range [0, 255].
    Args:
        img (numpy.ndarray): Image array to save.
        fpath (str): Path to save the image file.
    """
    img = (img*255).astype(np.uint8)
    io.imsave(fpath, img)
    print(f"Image saved to {fpath}")

def main(args):
    if args.model == "swinir":
        args_swinir = ArgsSWINIR(args.model_path)
        model = main_test_swinir.define_model(args_swinir)
        model.eval()
        model = model.to(device)
    else:
        model = None
    for fname in os.listdir(args.input_dir):
        print(f"Processing file: {fname}")
        if not (fname.endswith('.png') or fname.endswith('.jpg')):
            continue
        img_path = os.path.join(args.input_dir, fname)
        basename = os.path.splitext(fname)[0]
        mask_path = os.path.join(args.mask_dir, basename+ ".png")
        img = read_image(img_path)/255.
        mask = read_mask(mask_path)>0.5
        if args.model == "bm3d":
            denoised_img = bm3d_denoising(img, mask)

        elif args.model == "swinir":
            denoised_img = swinir_denoising(img, model)

        else:
            raise ValueError(f"Unsupported algorithm: {args.algorithm}")
        membrane_subracted, bg = subract_membrane(img, denoised_img, mask)
        output_path = os.path.join(args.output_dir_denoised, basename+ ".png")
        save_image(output_path, denoised_img)
        print(f"Denoised image saved to {output_path}")
        output_path = os.path.join(args.output_dir_membrane_subtracted, basename+ ".png")
        save_image(output_path, membrane_subracted)
        print(f"Membrane subtracted image saved to {output_path}")
        output_path = os.path.join(args.output_dir_bg, basename+ ".png")
        save_image(output_path,bg)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Denoise images using BM3D algorithm.")
    parser.add_argument("--input_dir", type=str,
                        default=r"/home/astar/Projects/membrane_detection/data/membrane/images/test",
                        help="Directory containing input images.")
    parser.add_argument("--mask_dir", type=str,
                        default=r"/home/astar/Projects/membrane_detection/data/membrane/labels/test",
                        help="Directory containing mask images.")
    parser.add_argument("--output_dir", type=str, default=r"/home/astar/Projects/vesicles_data/iclr_experiments/bm3d",
                        help="Directory to save denoised images.")
    parser.add_argument("--model", default="bm3d", type=str, help="algorithm to use for denoising, e.g., bm3d")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the pre-trained model (for swinir).")
    args = parser.parse_args()

    args.output_dir_denoised = os.path.join(args.output_dir, "denoised")
    args.output_dir_membrane_subtracted = os.path.join(args.output_dir, "membrane_subtracted")
    args.output_dir_bg = os.path.join(args.output_dir, "est_bg")
    os.makedirs(args.output_dir_denoised, exist_ok=True)
    os.makedirs(args.output_dir_membrane_subtracted, exist_ok=True)
    os.makedirs(args.output_dir_bg, exist_ok=True)
    main(args)




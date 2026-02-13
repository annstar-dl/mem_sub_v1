import argparse
import os
import numpy as np
from mrc_utils import downsample_micrograph
from sub_utils import read_parameters_from_yaml_file
from run_mrc_subtraction import read_img, read_mrc
from membrane_estimation import prepare_micrograph, fit_membrane
import matplotlib.pyplot as plt

def plot_loss_hist(loss_hist, file_name):
    """Plot the loss history and save the plot."""
    plt.figure()
    plt.plot(loss_hist, marker='o')
    plt.title(f'Loss History for {file_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Difference')
    plt.grid()
    plt.show()

def mean_squared_difference(img1, img2,mask):
    """Calculate the mean squared difference between two images."""
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    return np.mean(((img1 - img2)*mask) ** 2)


def process_file(args: argparse.Namespace):
    """Process a single MRC file to subtract membranes and save the results."""
    basename = os.path.splitext(args.file_name)[0] #file name without extension
    # read radius size from parameters file
    parameters = read_parameters_from_yaml_file()
    border = parameters["r"]  # Border size for fuzzy mask
    #read micrograph from mrc file
    img, header, voxel_size = read_mrc(os.path.join(args.imgs_path, args.file_name))
    #read downsampled membrane mask of a micrograph from png file
    mask = read_img(os.path.join(args.masks_path, basename + ".png"), True)
    if np.any(np.isnan(img)):
        raise ValueError(f"Empty file, Nan in {args.file_name}")
    # downsample the micrograph if needed
    img_ds = downsample_micrograph(img, voxel_size[0], border, "center")
    # check if the image is the same size as the mask
    if img_ds.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image {args.file_name} and mask {basename}.png must have the same dimensions. "
            f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    #read parameters from the YAML file
    parameters = read_parameters_from_yaml_file()
    # run membrane subtraction algorithm
    img, mask, row_idx, col_idx = prepare_micrograph(img, mask, border)
    loss_hist = []
    for i in range(5):
        parameters["nb_iter"]
        membrane = fit_membrane(img, mask, row_idx, col_idx, parameters)
        mse = mean_squared_difference(img, membrane, mask)
        loss_hist.append(mse)
    plot_loss_hist(loss_hist, args.file_name)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Run membrane subtraction on a single MRC file")
    args.add_argument("--imgs_path", type=str, help="Directory where the MRC files are located")
    args.add_argument("--masks_path", type=str, help="Directory where the membrane masks are located")
    args.add_argument("--file_name", type=str, default=None, help="Name of the MRC file to process")
    args = args.parse_args()
    process_file(args)
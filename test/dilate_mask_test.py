import torch
import pytest
from sampling_grid import dilate_mask, gaussian_filter, get_sampling_grid
from get_basis_test import visualize_3_images
import os
from read_matlab import load_image_from_mat
from matplotlib import pyplot as plt

def compare_dilated_masks():
    """
    Compare the mask extracted from the patch with the mask from MATLAB.
    Args:
        mask (torch.Tensor): Mask extracted from the patch.
        mask_matlab (np.ndarray): Mask from MATLAB.

    Returns:
        bool: True if the masks are similar, False otherwise.
    """
    maindir = r"/home/astar/Projects/data_from_matlab_code"
    file_path = os.path.join(maindir, r'mk_1.mat')
    mask = load_image_from_mat(file_path, ["mask"])
    mask = torch.tensor(mask, dtype=torch.float32)  # Convert mask to tensor
    mask_dilate = dilate_mask(mask,1)
    # Convert mask from PyTorch tensor to NumPy array
    fname =  os.path.join(maindir,"mk1_mask_dilate.mat")
    mask_dilate_matlab = load_image_from_mat(fname, "mask1")
    mask_dilate_matlab = torch.tensor(mask_dilate_matlab)
    # Check if the values are similar
    visualize_3_images(mask_dilate, mask, mask_dilate-mask_dilate_matlab,
                       title1="Mask dilated from PyTorch", title2="Mask ", title3="Mask dilate PyTorch - MATLAB")
def compare_d_times_dilated_masks():
    """
    Compare the mask extracted from the patch with the mask from MATLAB.
    Args:
        mask (torch.Tensor): Mask extracted from the patch.
        mask_matlab (np.ndarray): Mask from MATLAB.

    Returns:
        bool: True if the masks are similar, False otherwise.
    """
    maindir = r"/home/astar/Projects/data_from_matlab_code"
    file_path = os.path.join(maindir, r'mk_1.mat')
    mask = load_image_from_mat(file_path, ["mask"])
    mask = torch.tensor(mask, dtype=torch.float32)  # Convert mask to tensor
    mask_dilate = dilate_mask(mask, 1)
    for _ in range(4):
        mask_dilate = dilate_mask(mask_dilate, 1)
    # Convert mask from PyTorch tensor to NumPy array
    fname = os.path.join(maindir, "mk1_mask_dilated_d_times.mat")
    mask_dilate_matlab = load_image_from_mat(fname, "mask1")
    mask_dilate_matlab = torch.tensor(mask_dilate_matlab)
    # Check if the values are similar
    visualize_3_images(mask_dilate, mask_matlab, mask_dilate - mask_dilate_matlab,
                       title1="Mask dilated from PyTorch", title2="Mask Matlab", title3="Mask dilate PyTorch - MATLAB")
if __name__ == "__main__":

    #compare_dilated_masks()
    compare_d_times_dilated_masks()
    plt.show()
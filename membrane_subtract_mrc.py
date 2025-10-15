import copy

import numpy as np
import torch
from PIL import Image
import os
from sampling_grid import get_sampling_grid, select_points_within_boundary
from basis_fn import get_basis
from fit_basis_to_data import fit_basis_to_data_batched
from utils import read_dict_from_yaml_file
from bg_estimation import get_background

def add_border_to_mask(mask, border):
    mask_w_border = copy.deepcopy(mask)
    mask_w_border[:border, :] = 1
    mask_w_border[-border:, :] = 1
    mask_w_border[:, :border] = 1
    mask_w_border[:, -border:] = 1
    return mask_w_border

def  membrane_subtract(img,mask,border = 0):
    """    Subtract the membrane mask from the patch.
    Args:
        img (torch.Tensor): The micrograph(image with membranes) image tensor of shape (H, W).
        mask (torch.Tensor): The micrograph mask tensor of shape (H, W).
        border (int): The size of the border to be masked out for background estimation. Default is 0.
    Returns:
        torch.Tensor: The reconstructed membrane image, tensor of shape (H, W).
        torch.Tensor: The image with subtracted membranes, tensor of shape (H, W).
    """
    #estimate the background of the image
    if not img.dtype == np.float64:
        raise ValueError("Input image must be of type np.float64, got {}".format(img.dtype))
    if not mask.dtype == np.float64:
        raise ValueError("Mask must be of type np.float64, got {}".format(mask.dtype))
    # Check if the input image is 2D or 3D
    if img.ndim != 2:
        raise ValueError("Input image must be a 2D tensor, got {} dimensions".format(img.ndim))
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D tensor, got {} dimensions".format(mask.ndim))

    # read parameters from the YAML file
    parameters = read_dict_from_yaml_file()
    d = parameters["d"]  # Number of dilation iterations for the mask
    w = parameters["w"]  # Distance between grid points
    max_iter_gd = parameters["max_nb_iter_GD"]  # Maximum number of iterations for gradient descent
    rho = parameters["rho"]  # Learning rate for gradient descent
    r = parameters["r"]  # Radius of neighboring around grid point
    nb_iter = parameters["nb_iter"]  # Number of iterations for fitting
    # create mask with border to estimate the background
    if border>0:
        print("Masking the border of the image for background estimation")
        mask_with_border = add_border_to_mask(mask, border)
        bg, img = get_background(img, mask_with_border, sigma=30.0)
    else:
        bg, img = get_background(img, mask, sigma=30.0)
    # Convert numpy arrays to torch tensors
    img, mask = torch.tensor(img, dtype=torch.float64), torch.tensor(mask, dtype=torch.float64)

    # Get the sampling grid from the mask
    mask, row_idx, col_idx = get_sampling_grid(mask, d, w)
    # Ensure that the extracted patch points stay within the boundary of the mask
    row_idx, col_idx = select_points_within_boundary(img, r, row_idx,
                                                     col_idx)

    # Ensure the image is centered around zero
    ###!!!TODO: GET RID OF THE MEAN SUBTRACTION IN THE FUTURE OR ADD IT BACK AFTER SUBTRACTION
    #img = img - torch.mean(img)

    # Clone the original image to avoid modifying it
    dataimg = img.detach().clone()
    # Fit basis to the previous reconstruction to achieve better results
    for _ in range(nb_iter):
        basis = get_basis(dataimg, mask, row_idx, col_idx, r)
        imgout = fit_basis_to_data_batched(img,basis, row_idx, col_idx,r, rho, max_iter_gd,w)
        dataimg = imgout

    # Subtract the membrane from the original image
    imgout = imgout.to(mask.device)
    imgout = imgout * mask
    imgout = imgout.detach().cpu().numpy()
    return imgout
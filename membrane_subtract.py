import torch
from PIL import Image
import os
from sampling_grid import get_sampling_grid, select_points_within_boundary
from basis_fn import get_basis
from fit_basis_to_data import fit_basis_to_data, fit_basis_to_data_batched
from utils import read_dict_from_yaml_file


def  membrane_subtract(img, mask):
    """    Subtract the membrane mask from the patch.
    Args:
        img (torch.Tensor): The micrograph(image with membranes) image tensor of shape (H, W).
        mask (torch.Tensor): The micrograph mask tensor of shape (H, W).
    Returns:
        torch.Tensor: The patch with the membrane subtracted.
    """
    # Check if the input image is 2D or 3D
    if img.dim() != 2:
        raise ValueError("Input image must be a 2D tensor, got {} dimensions".format(img.dim()))
    if mask.dim() != 2:
        raise ValueError("Mask must be a 2D tensor, got {} dimensions".format(mask.dim()))
    img = img.to(torch.float64)
    img = img - torch.mean(img)
    #read parameters from the YAML file
    parameters = read_dict_from_yaml_file()
    d = parameters["d"]  # Number of dilation iterations for the mask
    w = parameters["w"]  # Distance between grid points
    max_iter_gd = parameters["max_nb_iter_GD"]  # Maximum number of iterations for gradient descent
    rho = parameters["rho"]  # Learning rate for gradient descent
    r = parameters["r"] # Radius of neighboring around grid point
    nb_iter = parameters["nb_iter"]  # Number of iterations for fitting

    mask, row_idx, col_idx = get_sampling_grid(mask, d, w)  # Get the sampling grid from the mask
    row_idx, col_idx = select_points_within_boundary(img, r, row_idx,
                                                     col_idx)

    dataimg = img.detach().clone()  # Clone the original image to avoid modifying it

    for _ in range(nb_iter):
        basis = get_basis(dataimg, mask, row_idx, col_idx, r)
        imgout = fit_basis_to_data_batched(img,basis, row_idx, col_idx,r, rho, max_iter_gd,w)
        dataimg = imgout

    # Subtract the membrane from the original image
    imgout = imgout.to(mask.device)
    subtracted_img = img - imgout * mask
    return imgout, subtracted_img
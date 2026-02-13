import copy
import numpy as np
import torch
from mem_sub.membrane_est.sampling_grid import get_sampling_grid, select_points_within_boundary
from mem_sub.membrane_est.basis_fn import get_basis
from mem_sub.membrane_est.fit_basis_to_data import fit_basis_to_data_batched
from mem_sub.membrane_est.utils import read_parameters_from_yaml_file
from mem_sub.membrane_est.bg_estimation import get_background

def add_border_to_mask(mask, border):
    """Add the border to the mask to compensate
    for zero padding during downsampling"""
    mask_w_border = copy.deepcopy(mask)
    mask_w_border[:border, :] = 1
    mask_w_border[-border:, :] = 1
    mask_w_border[:, :border] = 1
    mask_w_border[:, -border:] = 1
    return mask_w_border

def  prepare_micrograph(img, mask, parameters, border = 0):
    """    Subtract the membrane mask from the patch.
    Args:
        img (torch.Tensor): The micrograph(image with membranes) image tensor of shape (H, W).
        mask (torch.Tensor): The micrograph mask tensor of shape (H, W).
        parameters (dict): The parameters read from the YAML file.
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


    d = parameters["d"]  # Number of dilation iterations for the mask
    w = parameters["w"]  # Distance between grid points
    r = parameters["r"]  # Radius of neighboring around grid point
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
    # check if there are any grid points detected
    if row_idx.shape[0] == 0:
        raise ValueError("No grid points detected for sampling grid. "
                         "Please check the mask and input image.")
    # Ensure that the extracted patch points stay within the boundary of the mask
    row_idx, col_idx = select_points_within_boundary(img, r, row_idx, col_idx)
    # check if there are any grid points left after selecting within boundary
    if row_idx.shape[0] == 0:
        raise ValueError("No grid points left after selecting within boundary. "
                         "Please check the mask and input image.")
    return img, mask, row_idx, col_idx

def find_grid_angles(img, row_idx, col_idx, parameters):
    """
    Find the angles of the membrane profiles at each grid point.
    :param img: Torch.Tensor. Input image tensor of micrograph of shape (H, W).
    :param mask: Torch.Tensor. Smoothed segmentation mask tensor of shape (H, W).
    :param row_idx: Torch.Tensor. Y coordinates of grid of shape (N), number of grid points.
    :param col_idx: Torch.Tensor. X coordinates of grid of shape (N), number of grid points.
    :param parameters: dict. The parameters read from the YAML file.
    :return:
    torch.Tensor: The angles of the membrane profiles at each grid point of shape (N,).
    """
    # read parameters from the YAML file
    r = parameters["r"]  # Radius of neighboring around grid point
    dataimg = img.detach().clone()
    _, thetas = get_basis(dataimg, row_idx, col_idx, r)
    return thetas

def fit_membrane(img, mask, row_idx, col_idx,parameters):
    """
    Fit basis functions to estimate the membrane in the image.
    :param img: Torch.Tensor. Input image tensor of micrograph of shape (H, W).
    :param mask: Torch.Tensor. Smoothed segmentation mask tensor of shape (H, W).
    :param row_idx: Torch.Tensor. Y coordinates of grid of shape (N), number of grid points.
    :param col_idx: Torch.Tensor. X coordinates of grid of shape (N), number of grid points.
    :param parameters: dict. The parameters read from the YAML file.
    :return:
    np.ndarray: The estimated membrane image of shape (H, W).
    """
    w = parameters["w"]  # Distance between grid points
    max_iter_gd = parameters["max_nb_iter_GD"]  # Maximum number of iterations for gradient descent
    rho = parameters["rho"]  # Learning rate for gradient descent
    r = parameters["r"]  # Radius of neighboring around grid point
    nb_iter = parameters["nb_iter"]  # Number of iterations for fitting
    # Clone the original image to avoid modifying it
    dataimg = img.detach().clone()
    # Find the membrane using basis functions and fit it to the data
    # Fit basis to the previous reconstruction to achieve better results
    for _ in range(nb_iter):
        basis, _ = get_basis(dataimg, row_idx, col_idx, r)
        imgout = fit_basis_to_data_batched(img,basis, row_idx, col_idx,r, rho, max_iter_gd,w)
        dataimg = imgout

    # Smooth the edges of membrane estimate using the smoothed mask
    imgout = imgout.to(mask.device)
    imgout = imgout * mask
    imgout = imgout.detach().cpu().numpy()
    return imgout

def membrane_angle_estimation(img, mask, border = 0, return_theta = False, return_membrane = True):
    """
    Estimate the angles of the membrane profiles at each grid point.
    :param img: Torch.Tensor. Input image tensor of micrograph of shape (H, W).
    :param mask: Torch.Tensor. Binary segmentation mask tensor of shape (H, W).
    :param border: int. The size of the border to be masked out for background estimation. Default is 0.
    :param return_theta: bool. Whether to return the angles of the membrane profiles at each grid point. Default is False.
    :param return_membrane: bool. Whether to return the estimated membrane image. Default is True.
    :return:
    torch.Tensor: The angles of the membrane profiles at each grid point of shape (N,).
    """
    #read parameters from the YAML file
    parameters = read_parameters_from_yaml_file()
    # Prepare the micrograph and get the sampling grid
    img, mask, row_idx, col_idx = prepare_micrograph(img, mask, parameters, border)
    # Find the angles of the membrane profiles at each grid point
    if return_theta:
        angles = find_grid_angles(img, row_idx, col_idx, parameters)
        dict_angles = {'row_idx': row_idx.numpy(), 'col_idx': col_idx.numpy(), 'angles': angles.numpy()}
    else:
        dict_angles = None
    # Fit basis functions to estimate the membrane in the image
    if return_membrane:
        membrane = fit_membrane(img, mask, row_idx, col_idx, parameters)
    else:
        membrane = None
    return membrane, dict_angles






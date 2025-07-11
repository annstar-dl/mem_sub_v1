import torch
from basis_fn import create_gaussian_disc, get_radius_of_inner_circle, get_w_function
from utils import add_patches_to_image, get_patches_from_image_adv_indexing, creat_idx_batches_for_parl_sum, add_patches_to_image_batched



def fit_basis_to_data_batched(img, basis, row_idx, col_idx, r, rho, max_iter, step):
    """
    Fit bases to data using gradient descent method. Here we optimize alpha values
    for each basis function. Then we reconstruct the image using the optimized alpha
     values and basis functions.
    :param img: Input image of shape (H,W), where H, W are the height and width of the image.
    :param basis: Basis functions of shape (N, r, r), where N is the number of basis functions, r
    is an inner radius of the neighbourhood.
    :param row_idx: X coordinates of grid of shape (N,), where N is the number of grid points.
    :param col_idx: Y coordinates of grid of shape (N,), where N is the number of grid points.
    :param r: Radius of a neighbourhood.
    :param rho: Learning rate for gradient descent.
    :param max_iter: Maximum number of iterations for gradient descent.
    :param step: grid step size (stride).
    :return:
    """
    #Check if the input image is 2D
    if img.dim() != 2:
        raise ValueError("Input image must be a 2D tensor, got {} dimensions".format(img.dim()))
    img = img.to(basis.device)  # Move the image to the same device as the basis
    nGrid = row_idx.shape[0]  # Number of grid points
    gradient = torch.zeros((nGrid,),device =basis.device)  # Initialize the gradient tensor
    dataimg = img.detach().clone()  # Clone the original image to avoid modifying it
    imgout = torch.zeros_like(img)  # Initialize the output image
    r_in = get_radius_of_inner_circle(r)  # Radius of the inner circle
    # normalize the basis
    norms_of_basis = torch.linalg.norm(basis, dim=(1, 2), keepdim=True)  # return the norm of each basis function
    #norm_of_basis has shape (N, 1, 1)
    basis = basis / norms_of_basis  # Normalize the basis functions
    diff_norm = [torch.linalg.norm(img, dim=(0, 1), keepdim=True).squeeze()]
    stop_flag = 0
    iter=0
    batched_row_idxs, batched_col_idxs, bases_idxs = creat_idx_batches_for_parl_sum(row_idx,col_idx,r_in,step)

    while stop_flag == 0 and iter < max_iter:
        iter = iter + 1
        tmp_dataimg = get_patches_from_image_adv_indexing(dataimg, r_in, row_idx, col_idx)
        prod = tmp_dataimg * basis
        gradient = torch.sum(prod, [1,2])[...,None,None]

        imgout = add_patches_to_image_batched(rho*gradient*basis, imgout, r_in, batched_row_idxs, batched_col_idxs, bases_idxs)


        diff_norm.append(
            torch.linalg.norm(img - imgout)) # Compute the difference norm

        dataimg = img - imgout  # Update the data image with the output image
        # Stop condition: if the difference norm is less than a threshold
        if torch.abs(diff_norm[-2] - diff_norm[-1])/diff_norm[-2] <=1e-3:
            stop_flag = 1
    return imgout


def fit_basis_to_data(img, basis, row_idx, col_idx, r, rho, max_iter,step):
    """
    Fit bases to data using gradient descent method. Here we optimize alpha values
    for each basis function. Then we reconstruct the image using the optimized alpha
     values and basis functions.
    :param img: Input image of shape (H,W), where H, W are the height and width of the image.
    :param basis: Basis functions of shape (N, r, r), where N is the number of basis functions, r
    is an inner radius of the neighbourhood.
    :param row_idx: X coordinates of grid of shape (N,), where N is the number of grid points.
    :param col_idx: Y coordinates of grid of shape (N,), where N is the number of grid points.
    :param r: Radius of a neighbourhood.
    :param rho: Learning rate for gradient descent.
    :param max_iter: Maximum number of iterations for gradient descent.
    :param step: grid step size (stride).
    :return:

    """
    #Check if the input image is 2D
    if img.dim() != 2:
        raise ValueError("Input image must be a 2D tensor, got {} dimensions".format(img.dim()))
    img = img.to(basis.device)  # Move the image to the same device as the basis
    nGrid = row_idx.shape[0]  # Number of grid points
    gradient = torch.zeros((nGrid,),device =basis.device)  # Initialize the gradient tensor
    dataimg = img.detach().clone()  # Clone the original image to avoid modifying it
    imgout = torch.zeros_like(img)  # Initialize the output image
    r_in = get_radius_of_inner_circle(r)  # Radius of the inner circle
    # normalize the basis
    norms_of_basis = torch.linalg.norm(basis, dim=(1, 2), keepdim=True)  # return the norm of each basis function
    #norm_of_basis has shape (N, 1, 1)
    basis = basis / norms_of_basis  # Normalize the basis functions
    diff_norm = [torch.linalg.norm(img, dim=(0, 1), keepdim=True).squeeze()]
    stop_flag = 0
    iter=0

    while stop_flag == 0 and iter < max_iter:
        iter = iter + 1
        tmp_dataimg = get_patches_from_image_adv_indexing(dataimg, r_in, row_idx, col_idx)
        prod = tmp_dataimg * basis
        gradient = torch.sum(prod, [1,2])[...,None,None]

        imgout = add_patches_to_image(rho*gradient*basis, imgout, r_in, row_idx, col_idx)
        diff_norm.append(
            torch.linalg.norm(img - imgout)) # Compute the difference norm

        dataimg = img - imgout  # Update the data image with the output image
        # Stop condition: if the difference norm is less than a threshold
        if torch.abs(diff_norm[-2] - diff_norm[-1])/diff_norm[-2] <=1e-3:
            stop_flag = 1
    return imgout


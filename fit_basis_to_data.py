import torch
from basis_fn import create_gaussian_disc, get_radius_of_inner_circle, get_w_function

def fit_basis_to_data(img, basis, row_idx, col_idx, r):
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
    :return:
    """
    #Check if the input image is 2D
    if img.dim() != 2:
        raise ValueError("Input image must be a 2D tensor, got {} dimensions".format(img.dim()))
    nGrid = row_idx.shape[0]  # Number of grid points
    gradient = torch.zeros((nGrid,))
    dataimg = img.clone()  # Clone the original image to avoid modifying it
    imgout = torch.zeros_like(img)  # Initialize the output image
    r_in = get_radius_of_inner_circle(r)  # Radius of the inner circle
    rho = 0.05  # Learning rate for gradient descent
    # normalize the basis
    norms_of_basis = torch.linalg.norm(basis, dim=(1, 2), keepdim=True)  # return the norm of each basis function
    #norm_of_basis has shape (N, 1, 1)
    basis = basis / norms_of_basis  # Normalize the basis functions

    diff_norm = [torch.linalg.norm(img, dim=(0, 1), keepdim=True).squeeze()]
    max_iter = 30  # Maximum number of iterations for gradient descent
    stop_flag = 0
    iter=0

    while stop_flag == 0 and iter < max_iter:
        iter = iter + 1
        for i in range(nGrid):
            b = basis[i]  # Get the basis function for the current grid point
            tmp = dataimg[row_idx[i] - r_in:row_idx[i] + r_in + 1, col_idx[i] - r_in:col_idx[i] + r_in + 1]  # Extract the neighborhood around the grid point
            # Compute the gradient for the current grid point
            gradient[i] = torch.sum(tmp * b)
        # Update imgout
        for i in range(nGrid):
            # Update the output image using the gradient and basis function
            imgout[row_idx[i] - r_in:row_idx[i] + r_in + 1, col_idx[i] - r_in:col_idx[i] + r_in + 1] += rho * gradient[i] * basis[i]

        diff_norm.append(
            torch.linalg.norm(img - imgout, dim=(0, 1), keepdim=True).squeeze()) # Compute the difference norm
        dataimg = img - imgout  # Update the data image with the output image
        # Stop condition: if the difference norm is less than a threshold
        if torch.abs(diff_norm[-2] - diff_norm[-1])/diff_norm[-2] <=1e-3:
            stop_flag = 1

    return imgout
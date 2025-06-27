import time
import torch
from basis_fn import create_gaussian_disc, get_radius_of_inner_circle, get_w_function
from utils import get_patches_from_image, add_patches_to_image, get_patches_from_image_adv_indexing, add_patches_to_image_adv_indexing


def fill_tensor_with_bases(basis,img, row_idx, col_idx, r_in,cur_idx,elem_in_batch):
    bases_batch = torch.zeros(elem_in_batch, img.shape[-2], img.shape[-1],dtype = img.dtype)  # Initialize the full basis tensor
    row_starts = row_idx[cur_idx:cur_idx+elem_in_batch] - r_in
    col_starts = col_idx[cur_idx:cur_idx+elem_in_batch] - r_in
    patch_size = 2 * r_in + 1
    rows = torch.arange(patch_size, device=basis.device)
    cols = torch.arange(patch_size, device=basis.device)
    grid_rows = torch.zeros(elem_in_batch * patch_size**2, dtype=torch.int64, device=basis.device)  # Initialize the grid rows
    grid_cols = torch.zeros(elem_in_batch * patch_size**2, dtype=torch.int64, device=basis.device)
    for i in range(elem_in_batch):
        r,c = row_starts[i], col_starts[i]
        r,c = torch.meshgrid([r,c])
        grid_rows[i*patch_size**2:i*patch_size**2+patch_size**2]=r.flatten()
        grid_cols[i*patch_size**2:i*patch_size**2+patch_size**2]=c.flatten()
    batch_indices = torch.arange(elem_in_batch).unsqueeze(-1).expand(-1,patch_size**2).flatten()  # Indices for the batch dimension
    bases_batch[batch_indices, grid_rows, grid_cols] = basis[cur_idx:cur_idx + elem_in_batch].flatten()
    return bases_batch

def fill_tensor_with_bases_seq(basis,img, row_idx, col_idx, r_in,cur_idx,elem_in_batch):
    bases_batch = torch.zeros(elem_in_batch, img.shape[-2], img.shape[-1],dtype = img.dtype)  # Initialize the full basis tensor
    for i in range(elem_in_batch):
        bases_batch[j, row_idx[j]-r_in:row_idx[j]+r_in+1, col_idx[j]-r_in:col_idx[j]+r_in+1] = basis[j]
    return bases_batch

def fit_basis_to_data_matrix(img, basis, row_idx, col_idx, r, rho, max_iter):
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
    batch_size = 100  # Define the batch size for processing
    diff_norm = [torch.linalg.norm(img, dim=(0, 1), keepdim=True).squeeze()]
    stop_flag = 0
    iter=0

    while stop_flag == 0 and iter < max_iter:
        iter = iter + 1
        for i in range(0,nGrid,batch_size):
            elem_in_batch = min(batch_size, len(basis[i:]))  # Number of elements in the current batch
            #basis_full = torch.zeros(elem_in_batch, img.shape[-2], img.shape[-1])  # Initialize the full basis tensor
            #for j in range(elem_in_batch):
            #    basis_full[j, row_idx[j]-r_in:row_idx[j]+r_in+1, col_idx[j]-r_in:col_idx[j]+r_in+1] = basis[j]  # Fill the full basis tensor with the basis functions at the grid points
            bases_batch = fill_tensor_with_bases(basis, img, row_idx, col_idx, r_in, i, elem_in_batch)  # Fill the batch basis tensor with the basis functions at the grid points
            prod = dataimg.unsqueeze(0) * bases_batch
            gradient[i:i+elem_in_batch] = torch.sum(prod, [1,2])
        gradient = gradient[...,None,None]  # Compute the gradient by summing over the basis functions
        imgout_delta = torch.zeros_like(imgout) # Initialize the output image delta tensor
        for i in range(0,nGrid,batch_size):
            elem_in_batch = min(batch_size, len(basis[i:]))  # Number of elements in the current batch
            bases_batch = fill_tensor_with_bases(basis, img, row_idx, col_idx, r_in, i, elem_in_batch)  # Fill the batch basis tensor with the basis functions at the grid points
            #basis_full = torch.zeros(elem_in_batch, img.shape[-2], img.shape[-1])  # Initialize the full basis tensor
            #for j in range(elem_in_batch):
            #    basis_full[j, row_idx[j]-r_in:row_idx[j]+r_in+1, col_idx[j]-r_in:col_idx[j]+r_in+1] = basis[j]
            imgout_delta_tmp = bases_batch*gradient[i:i+elem_in_batch]
            imgout_delta += rho*torch.sum(imgout_delta_tmp,0)
        imgout = imgout + imgout_delta  # Update the output image by adding the gradient scaled by the learning rate

        diff_norm.append(
            torch.linalg.norm(img - imgout)) # Compute the difference norm

        dataimg = img - imgout  # Update the data image with the output image
        # Stop condition: if the difference norm is less than a threshold
        if torch.abs(diff_norm[-2] - diff_norm[-1])/diff_norm[-2] <=1e-3:
            stop_flag = 1
    return imgout

def fit_basis_to_data(img, basis, row_idx, col_idx, r, rho, max_iter):
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

        imgout = add_patches_to_image_adv_indexing(rho*gradient*basis, imgout, r_in, row_idx, col_idx)


        diff_norm.append(
            torch.linalg.norm(img - imgout)) # Compute the difference norm

        dataimg = img - imgout  # Update the data image with the output image
        # Stop condition: if the difference norm is less than a threshold
        if torch.abs(diff_norm[-2] - diff_norm[-1])/diff_norm[-2] <=1e-3:
            stop_flag = 1
    return imgout

def fit_basis_to_data_unfold(img, basis, row_idx, col_idx, r, rho, max_iter):
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
        tmp_dataimg = get_patches_from_image_unfold(dataimg, r_in, 4, row_idx, col_idx).squeeze().to(basis.device)
        prod = tmp_dataimg * basis
        gradient = torch.sum(prod, [1,2])[...,None,None]
        imgout = add_patches_to_image_adv_indexing(rho * gradient * basis, imgout, r_in, 4, row_idx, col_idx)
        diff_norm.append(
            torch.linalg.norm(img - imgout)) # Compute the difference norm

        dataimg = img - imgout  # Update the data image with the output image
        # Stop condition: if the difference norm is less than a threshold
        if torch.abs(diff_norm[-2] - diff_norm[-1])/diff_norm[-2] <=1e-3:
            stop_flag = 1
    return imgout
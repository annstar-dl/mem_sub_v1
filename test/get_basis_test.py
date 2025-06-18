import matplotlib.pyplot as plt
from extract_small_patch import extract_small_patch_with_mask
from sampling_grid import get_sampling_grid
from basis_fn import get_basis, get_basis_sequential
from align_image_test import visualize_3_images, visualize_2_images
from sampling_grid_test import visualize_sampling_grid
from sampling_grid import select_points_within_boundary
from basis_fn import create_gaussian_disc, get_radius_of_inner_circle
import torch
import time

def get_basis_parallelized_test(patch_size, left, top):
    """Test the get_bases function with a small patch.
    This function extracts a small patch from the image, applies a sampling grid,
    and reconstructs the patch using the basis functions.
    Args:
        patch_size (tuple): Size of the patch to extract (width, height).
        left (int): Left coordinate of the patch.
        top (int): Top coordinate of the patch.
    Returns:
        torch.Tensor: The reconstructed patch tensor.
        torch.Tensor: The basis functions used for reconstruction.
        torch.Tensor: The angles used for rotation.
    """

    grid_size = 2  # Size of the grid for sampling
    stride = 4  # Stride for the sampling grid
    # Extract a small patch and its mask
    patch_tensor, mask_tensor = extract_small_patch_with_mask(left, top, patch_size)
    patch_tensor = patch_tensor[0]
    mask_tensor, row_idx, col_idx = get_sampling_grid(mask_tensor, grid_size, stride)
    r = 14  # Radius of neighboring around grid point
    row_idx, col_idx = select_points_within_boundary(patch_tensor, r, row_idx,
                                                     col_idx)  # Select points within the boundary of the image based on the radius
    # Visualize the extracted patch and mask and the sampling grid
    visualize_sampling_grid(mask_tensor, row_idx, col_idx)

    cntr = r + 1  # Center of the neighborhood
    r_in = get_radius_of_inner_circle(r)  # Radius of the inner circle
    binaryImage, gaussWt = create_gaussian_disc(2 * [2 * r_in + 1], r_in)  # Create a binary disc and Gaussian weights

    dataimg = patch_tensor.clone()  # Clone the original image to avoid modifying it
    nGrid = row_idx.shape[0]  # Number of grid points
    # Initialize the output image and weight image
    imgout = torch.zeros_like(patch_tensor)  # Initialize the output image
    wtimg = torch.zeros_like(patch_tensor)  # Initialize the weight image
    basis, theta = get_basis(patch_tensor, dataimg, mask_tensor, row_idx, col_idx, r, True)
    for i in range(nGrid):
        imgout[row_idx[i] - r_in:row_idx[i] + r_in + 1, col_idx[i] - r_in:col_idx[i] + r_in + 1] += basis[i]
        wtimg[row_idx[i] - r_in:row_idx[i] + r_in + 1, col_idx[i] - r_in:col_idx[i] + r_in + 1] += gaussWt
    imgout = imgout / (wtimg + 1e-6)  # Normalize the output image by the weights
    visualize_3_images(patch_tensor, imgout, wtimg,
                       title1="Original Patch",
                       title2="Reconstructed Patch",
                       title3="Weight Image")
    return imgout, basis, theta

def get_basis_sequential_test(patch_size, left, top):
    """Test the get_bases function with a small patch.
    This function extracts a small patch from the image, applies a sampling grid,
    and reconstructs the patch using the basis functions.
    Args:
        patch_size (tuple): Size of the patch to extract (width, height).
        left (int): Left coordinate of the patch.
        top (int): Top coordinate of the patch.
    Returns:
        torch.Tensor: The reconstructed patch tensor.
        torch.Tensor: The basis functions used for reconstruction.
        torch.Tensor: The angles used for rotation.
    """
    grid_size = 2  # Size of the grid for sampling
    stride = 4  # Stride for the sampling grid
    # Extract a small patch and its mask
    patch_tensor, mask_tensor = extract_small_patch_with_mask(left, top, patch_size)
    patch_tensor = patch_tensor[0]
    mask_tensor, row_idx, col_idx = get_sampling_grid(mask_tensor, grid_size, stride)
    r = 14  # Radius of neighboring around grid point
    row_idx, col_idx = select_points_within_boundary(patch_tensor, r, row_idx,
                                         col_idx)  # Select points within the boundary of the image based on the radius
    # Visualize the extracted patch and mask and the sampling grid
    #visualize_sampling_grid(mask_tensor, row_idx, col_idx)

    cntr = r + 1  # Center of the neighborhood
    r_in = get_radius_of_inner_circle(r)  # Radius of the inner circle
    binaryImage, gaussWt = create_gaussian_disc(2*[2 * r_in+1], r_in)  # Create a binary disc and Gaussian weights

    dataimg = patch_tensor.clone()  # Clone the original image to avoid modifying it
    nGrid = row_idx.shape[0]  # Number of grid points
    # Initialize the output image and weight image
    imgout = torch.zeros_like(patch_tensor)  # Initialize the output image
    wtimg = torch.zeros_like(patch_tensor)  # Initialize the weight image
    basis, theta = get_basis_sequential(patch_tensor, dataimg, mask_tensor, row_idx, col_idx, r, True)
    for i in range(nGrid):
        imgout[row_idx[i] - r_in:row_idx[i] + r_in+1, col_idx[i] - r_in:col_idx[i] + r_in+1] += basis[i]
        wtimg[row_idx[i] - r_in:row_idx[i] + r_in+1, col_idx[i] - r_in:col_idx[i] + r_in+1] += gaussWt
    imgout = imgout / (wtimg+1e-6) # Normalize the output image by the weights
    visualize_3_images(patch_tensor, imgout, wtimg,
                       title1="Original Patch",
                       title2="Reconstructed Patch",
                       title3="Weight Image")
    return imgout, basis, theta


if __name__ =="__main__":
    left = 0
    top = 0
    patch_size = None  # Width, Height
    start_seq = time.time()
    imgout_seq, basis_seq, theta_seq = get_basis_sequential_test(patch_size,left, top)
    end_seq = time.time()
    print("Time taken for sequential reconstruction:", end_seq - start_seq)
    start_paral = time.time()
    imgout_paral,basis_paral, theta_paral = get_basis_parallelized_test(patch_size, left, top)
    end_paral = time.time()
    print("Time taken for parallelized reconstruction:", end_paral - start_paral)
    print("Difference between sequential and parallelized reconstruction:",
          torch.linalg.norm(basis_seq - basis_paral,dim=[1,2]))


    plt.figure(figsize=(10, 5))
    plt.imshow(imgout_seq-imgout_paral, cmap='gray')
    plt.title("Difference between Sequential and Parallelized Bases")

    plt.show()
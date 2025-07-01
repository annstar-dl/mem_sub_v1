import matplotlib.pyplot as plt
from extract_small_patch import extract_small_patch_with_mask
from sampling_grid import get_sampling_grid, gaussian_filter, dilate_mask
from basis_fn import get_basis
from align_image_test import visualize_3_images, visualize_2_images,visualize_im
from sampling_grid_test import visualize_sampling_grid
from sampling_grid import select_points_within_boundary
from basis_fn import create_gaussian_disc, get_radius_of_inner_circle
import torch
from fit_basis_to_data import fit_basis_to_data
import time
from read_matlab import load_image_from_mat


def fit_basis_to_data_test(patch_size, left, top):
    """ Test the fit_basis_to_data function with a small patch."""

    grid_size = 2  # Size of the grid for sampling
    stride = 4  # Stride for the sampling grid
    # Extract a small patch and its mask
    patch_tensor, mask_tensor = extract_small_patch_with_mask(left, top, patch_size)
    mask_tensor = dilate_mask(mask_tensor,(3,3),(1.,1.))
    patch_tensor = patch_tensor[0]
    patch_tensor_f = gaussian_filter(patch_tensor, (15,15),(30,30))
    patch_tensor = patch_tensor - patch_tensor_f
    #check if device has gpu

    patch_tensor = patch_tensor - torch.mean(patch_tensor)  # Center the patch around zero
    mask_tensor, row_idx, col_idx = get_sampling_grid(mask_tensor, grid_size, stride)
    r = 14  # Radius of neighboring around grid point
    row_idx, col_idx = select_points_within_boundary(patch_tensor, r, row_idx,
                                         col_idx)  # Select points within the boundary of the image based on the radius
    # Visualize the extracted patch and mask and the sampling grid
    visualize_sampling_grid(mask_tensor, row_idx, col_idx)

    r_in = get_radius_of_inner_circle(r)  # Radius of the inner circle

    dataimg = patch_tensor.clone()  # Clone the original image to avoid modifying it
    # Initialize the output image and weight image
    imgout = torch.zeros_like(patch_tensor)  # Initialize the output image
    for iter in range(3):
        start_time = time.time()
        basis = get_basis(patch_tensor, dataimg, mask_tensor, row_idx, col_idx, r)
        print("Time taken to get basis rotation: {:.2f} seconds".format(time.time() - start_time))
        # Fit basis to data

        start_time = time.time()
        imgout = fit_basis_to_data(patch_tensor, basis, row_idx, col_idx, r)
        print("Time taken to fit basis to data: {:.2f} seconds".format(time.time() - start_time))
        visualize_2_images(patch_tensor, imgout,
                           title1="Original Patch",
                           title2="Reconstructed Patch iter{}".format(iter))
        dataimg = imgout
        # Visualize subtracted image
        subtracted_img = patch_tensor - imgout*mask_tensor
        visualize_2_images(patch_tensor, subtracted_img,
                           title1="Original Patch",
                           title2="Subtracted Patch")


def fit_basis_to_matlab_data_test():
    """ Test the fit_basis_to_data function with a small patch."""

    grid_size = 2  # Size of the grid for sampling
    stride = 4  # Stride for the sampling grid
    # Extract a small patch and its mask
    file_path = r'/home/astar/Projects/data_from_matlab_code/mk_1.mat'
    patch = load_image_from_mat(file_path,"img")
    mask = load_image_from_mat(file_path, "mask")
    patch_tensor = torch.tensor(patch, dtype=torch.float64)
    mask_tensor = torch.tensor(mask, dtype=torch.float32)
    #check if device has gpu
    patch_tensor_f = gaussian_filter(patch_tensor, (101,101),(30,30))
    #patch_tensor = patch_tensor - patch_tensor_f
    patch_tensor = patch_tensor - torch.mean(patch_tensor)  # Center the patch around zero

    mask_tensor, row_idx, col_idx = get_sampling_grid(mask_tensor, grid_size, stride)
    r = 14  # Radius of neighboring around grid point
    row_idx, col_idx = select_points_within_boundary(patch_tensor, r, row_idx,
                                         col_idx)  # Select points within the boundary of the image based on the radius
    # Visualize the extracted patch and mask and the sampling grid
    visualize_sampling_grid(mask_tensor, row_idx, col_idx)

    r_in = get_radius_of_inner_circle(r)  # Radius of the inner circle

    dataimg = patch_tensor.clone()  # Clone the original image to avoid modifying it
    # Initialize the output image and weight image
    imgout = torch.zeros_like(patch_tensor)  # Initialize the output image
    for iter in range(1):
        start_time = time.time()
        basis = get_basis(patch_tensor, dataimg, mask_tensor, row_idx, col_idx, r)
        print("Time taken to get basis rotation: {:.2f} seconds".format(time.time() - start_time))
        # Fit basis to data

        start_time = time.time()
        imgout = fit_basis_to_data(patch_tensor, basis, row_idx, col_idx, r)
        print("Time taken to fit basis to data: {:.2f} seconds".format(time.time() - start_time))
        dataimg = imgout
        # Visualize subtracted image
        subtracted_img = patch_tensor - imgout*mask_tensor
        visualize_3_images(patch_tensor,imgout, subtracted_img,
                           title1="Original Patch",
                           title2="Reconstructed Patch iter{}".format(iter),
                           title3="Subtracted Patch")
        visualize_2_images(patch_tensor,subtracted_img,"Original Patch", "Subtracted image")
        visualize_im(patch_tensor, "Original patch")
        visualize_im(subtracted_img, "Subtracted image")


if __name__ =="__main__":
    left = 0
    top = 0
    patch_size = None  # Width, Height
    fit_basis_to_matlab_data_test()
    plt.show()
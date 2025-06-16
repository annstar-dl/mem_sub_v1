import matplotlib.pyplot as plt
from extract_small_patch import extract_small_patch_with_mask
from sampling_grid import get_sampling_grid
from basis_fn import get_basis
from align_image_test import visualize_3_images, visualize_2_images
from sampling_grid_test import visualize_sampling_grid
from sampling_grid import select_points_within_boundary
from basis_fn import create_gaussian_disc, get_radius_of_inner_circle
import torch
from fit_basis_to_data import fit_basis_to_data



def fit_basis_to_data_test():
    """ Test the fit_basis_to_data function with a small patch."""
    left = 150
    top = 20
    patch_size = (160, 160)  # Width, Height
    grid_size = 2  # Size of the grid for sampling
    stride = 4  # Stride for the sampling grid
    # Extract a small patch and its mask
    patch_tensor, mask_tensor = extract_small_patch_with_mask(left, top, patch_size)
    patch_tensor = patch_tensor[0]
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
        basis = get_basis(patch_tensor, dataimg, mask_tensor, row_idx, col_idx, r)
        # Fit basis to data
        imgout = fit_basis_to_data(patch_tensor, basis, row_idx, col_idx, r)
        visualize_2_images(patch_tensor, imgout,
                           title1="Original Patch",
                           title2="Reconstructed Patch iter{}".format(iter))
        dataimg = imgout

        # Visualize subtracted image
        subtracted_img = patch_tensor - imgout
        visualize_2_images(patch_tensor, subtracted_img,
                           title1="Original Patch",
                           title2="Subtracted Patch")
if __name__ =="__main__":
    fit_basis_to_data_test()
    plt.show()
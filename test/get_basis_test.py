import matplotlib.pyplot as plt

from extract_small_patch import extract_small_patch_with_mask
from sampling_grid import get_sampling_grid
from basis_fn import get_basis
from align_image_test import visualize_3_images
from sampling_grid_test import visualize_sampling_grid
from sampling_grid import select_points_within_boundary
from basis_fn import create_gaussian_disc, get_radius_of_inner_circle
import torch

def get_basis_test():
    """
    Test the get_bases function with a small patch.
    This function is a placeholder and should be implemented based on specific requirements.
    """

    left = 150
    top = 20
    patch_size = (160, 160)  # Width, Height
    grid_size = 2  # Size of the grid for sampling
    stride = 4  # Stride for the sampling grid
    # Extract a small patch and its mask
    patch_tensor, mask_tensor = extract_small_patch_with_mask(left, top, patch_size)
    patch_tensor = patch_tensor[0]
    mask_tensor, x, y = get_sampling_grid(mask_tensor, grid_size, stride)
    r = 14  # Radius of neighboring around grid point
    x, y = select_points_within_boundary(patch_tensor, r, x,
                                         y)  # Select points within the boundary of the image based on the radius
    # Visualize the extracted patch and mask and the sampling grid
    visualize_sampling_grid(mask_tensor, x, y)

    cntr = r + 1  # Center of the neighborhood
    r_in = get_radius_of_inner_circle(r)  # Radius of the inner circle
    binaryImage, gaussWt = create_gaussian_disc(2 * r_in+1, r_in)  # Create a binary disc and Gaussian weights

    dataimg = patch_tensor.clone()  # Clone the original image to avoid modifying it
    nGrid = x.shape[0]  # Number of grid points
    for _ in range(3):
        imgout = torch.zeros_like(patch_tensor)  # Initialize the output image
        wtimg = torch.zeros_like(patch_tensor)  # Initialize the weight image
        basis = get_basis(patch_tensor, dataimg, mask_tensor, x, y, r)
        for i in range(nGrid):
            xi = x[i]
            yi = y[i]
            imgout[yi - r_in:yi + r_in+1, xi - r_in:xi + r_in+1] += basis[i]
            wtimg[yi - r_in:yi + r_in+1, xi - r_in:xi + r_in+1] += gaussWt
        imgout = imgout / (wtimg+1e-6) # Normalize the output image by the weights
        dataimg = imgout
        visualize_3_images(patch_tensor, imgout, wtimg,
                           title1="Original Patch",
                           title2="Reconstructed Patch",
                           title3="Weight Image")


if __name__ =="__main__":
    get_basis_test()
    plt.show()
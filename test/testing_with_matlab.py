import scipy.io
import numpy as np
import torch
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


def load_image_from_mat(file_path, variable_name='image'):
    """
    Load an image from a .mat file.

    Args:
        file_path (str): Path to the .mat file.
    Returns:
        np.ndarray: The loaded image as a NumPy array.
    """
    mat_data = scipy.io.loadmat(file_path)
    if variable_name not in mat_data:
        raise KeyError(f"Variable '{variable_name}' not found in the .mat file.")
    return mat_data[variable_name]

def compare_basis(basis, basis_matlab):
    """
    Compare the basis extracted from the patch with the basis from MATLAB.

    Args:
        basis (torch.Tensor): Basis extracted from the patch.
        basis_matlab (np.ndarray): Basis from MATLAB.

    Returns:
        bool: True if the bases are similar, False otherwise.
    """
    # Convert basis from PyTorch tensor to NumPy array
    basis_numpy = basis.numpy()
    # Check if the shapes match
    if basis_numpy.shape != basis_matlab.shape:
        return False
    # Check if the values are similar
    print("Calculated bases is close to MATLAB bases: ", np.allclose(basis_numpy, basis_matlab, atol=1e-6))
    basis_matlab = torch.tensor(basis_matlab)
    visualize_3_images(basis[0], basis_matlab[0], basis_matlab[0] - basis[0],
                       title1="Basis from PyTorch",
                       title2="Basis from MATLAB",
                       title3="Difference Basis")
    plt.show()

def compare_masks(mask, mask_matlab):
    """
    Compare the mask extracted from the patch with the mask from MATLAB.
    Args:
        mask (torch.Tensor): Mask extracted from the patch.
        mask_matlab (np.ndarray): Mask from MATLAB.

    Returns:
        bool: True if the masks are similar, False otherwise.
    """
    # Convert mask from PyTorch tensor to NumPy array
    mask_matlab = torch.tensor(mask_matlab)
    diff_mask = mask_matlab - mask
    visualize_3_images(mask, mask_matlab,diff_mask,"Mask from PyTorch", "Mask from MATLAB","Difference Mask")
    plt.show()


if __name__ == "__main__":
    # Example usage
    file_path = r'/home/astar/Projects/matlab_code/mem_data_1.mat'  # Replace with your .mat file path
    img = load_image_from_mat(file_path,"img")
    mask = load_image_from_mat(file_path,"mask") # Assuming 'mask' is the key for the mask in the .mat file
    file_path = r'/home/astar/Projects/data_from_matlab_code/basis_test.mat'  # Replace with your .mat file path
    basis_matlab = load_image_from_mat(file_path,"basis") # Assuming 'basis' is the key for the basis in the .mat file
    file_path = r'/home/astar/Projects/data_from_matlab_code/mask_test.mat'  # Replace with your .mat file path
    mask_matlab = load_image_from_mat(file_path,"mask1") # Assuming 'mask' is the key for the mask in the .mat file
    file_path = r'/home/astar/Projects/data_from_matlab_code/x_test.mat'  # Replace with your .mat file path
    x_matlab = load_image_from_mat(file_path,"x") # Assuming 'x' is the key for the x coordinates in the .mat file
    file_path = r'/home/astar/Projects/data_from_matlab_code/y_test.mat'  # Replace with your .mat file path
    y_matlab = load_image_from_mat(file_path,"y") # Assuming 'y' is the key for the y coordinates in the .mat file

    # Convert the mask and image to PyTorch tensors
    img = img - np.mean(img)  # Normalize the image by subtracting the mean
    mask_tensor = torch.tensor(mask, dtype=torch.float32)  # Convert mask to tensor
    patch_tensor = torch.tensor(img, dtype=torch.float32)  # Convert image to tensor
    grid_size = 2  # Size of the grid for sampling
    stride = 4  # Stride for the sampling grid
    # Extract a small patch and its mask

    mask_tensor, x, y = get_sampling_grid(mask_tensor, grid_size, stride)
    # substitute the mask and grid coordinates with the one from MATLAB
    #mask_tensor = torch.tensor(mask_matlab)
    visualize_2_images(mask_tensor, torch.tensor(mask_matlab),"Mask from PyTorch", "Mask from MATLAB")
    plt.show()
    row_idx = torch.tensor(x_matlab)-1
    col_idx = torch.tensor(y_matlab)-1
    #compare_masks(mask_tensor, mask_matlab)
    r = 14  # Radius of neighboring around grid point
    #x, y = select_points_within_boundary(patch_tensor, r,x,y)  # Select points within the boundary of the image based on the radius
    # Visualize the extracted patch and mask and the sampling grid
    visualize_sampling_grid(mask_tensor, row_idx, col_idx)

    r_in = get_radius_of_inner_circle(r)  # Radius of the inner circle

    dataimg = patch_tensor.clone()  # Clone the original image to avoid modifying it
    # Initialize the output image and weight image
    imgout = torch.zeros_like(patch_tensor)  # Initialize the output image
    for iter in range(1):
        basis = get_basis(patch_tensor, dataimg, mask_tensor, row_idx, col_idx, r)
        compare_basis(basis,basis_matlab)
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

    plt.show()
import scipy.io
import numpy as np
import torch
import matplotlib.pyplot as plt
from extract_small_patch import extract_small_patch_with_mask
from sampling_grid import get_sampling_grid, dilate_mask
from basis_fn import get_basis, get_basis_sequential
from align_image_test import visualize_3_images, visualize_2_images
from sampling_grid_test import visualize_sampling_grid
from sampling_grid import select_points_within_boundary
from basis_fn import create_gaussian_disc, get_radius_of_inner_circle
import torch
from fit_basis_to_data import fit_basis_to_data




def compare_angles(angles, angles_matlab):
    angles_matlab = torch.tensor(angles_matlab).squeeze().to(torch.float32)
    if angles.shape!=angles_matlab.shape:
        raise Exception(f"Shape tensors array from matlab and pytorch are not equal, "
                      f"and are matlab {angles_matlab.shape},"
                      f"and torch {angles.shape}")
    if not torch.allclose(angles, angles_matlab, atol=1e-6):
        angles_diff = torch.abs(angles - angles_matlab)
        angles_diff_binary = angles_diff > 1e-6
        nb_diff_angles = torch.sum(angles_diff_binary)
        print("Amount of angles that are different: ", nb_diff_angles)
        idx_max_diff = torch.argmax(angles_diff)
        raise Exception(f"Matlab angles are different from pytorch, example idx {idx_max_diff} pytorch {angles[idx_max_diff]} matlab {angles_matlab[idx_max_diff]} ")


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
    basis_matlab = torch.tensor(basis_matlab)
    # Check if the values are similar
    diff_norm = torch.linalg.norm(basis - basis_matlab, dim=(1, 2))
    idx_max = torch.argmax(diff_norm)

    visualize_3_images(basis[idx_max], basis_matlab[idx_max], basis_matlab[idx_max] - basis[idx_max],
                       title1="Basis from PyTorch",
                       title2="Basis from MATLAB",
                       title3=f"Difference norm {diff_norm[idx_max]}")


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
    visualize_3_images(mask, mask_matlab,diff_mask,"Mask from PyTorch", "Mask from MATLAB","Matlab - PyTorch")

def compare_reconstr_image(imgout, imgout_matlab):
    """Compare """
    imgout_matlab = torch.tensor(imgout_matlab)
    print("Norm of difference of reconstructed images: ", torch.linalg.norm(imgout - imgout_matlab))
    visualize_3_images(imgout, imgout_matlab, imgout - imgout_matlab, "Imgout Pytorch", "Imgout Matlab", "Pytorch - Matlab")

def compare_dilation(dilate_mask, dilate_mask_matlab):
    """
    Compare the dilated mask extracted from the patch with the dilated mask from MATLAB.
    Args:
        dilate_mask (torch.Tensor): Dilated mask extracted from the patch.
        dilate_mask_matlab (np.ndarray): Dilated mask from MATLAB.

    Returns:
        bool: True if the dilated masks are similar, False otherwise.
    """
    # Convert dilated mask from PyTorch tensor to NumPy array
    dilate_mask_matlab = torch.tensor(dilate_mask_matlab)
    diff_dilate_mask = dilate_mask_matlab - dilate_mask
    visualize_3_images(dilate_mask, dilate_mask_matlab, diff_dilate_mask,
                       title1="Dilated Mask from PyTorch",
                       title2="Dilated Mask from MATLAB",
                       title3="Dilated Mask Matlab - PyTorch")


if __name__ == "__main__":
    # Example usage
    file_path = r'/home/astar/Projects/matlab_code/mem_data_1.mat'  # Replace with your .mat file path
    img = load_image_from_mat(file_path,"img")
    mask = load_image_from_mat(file_path,"mask") # Assuming 'mask' is the key for the mask in the .mat file

    file_path = r'/home/astar/Projects/data_from_matlab_code/basis_test.mat'  # Replace with your .mat file path
    basis_matlab = load_image_from_mat(file_path,"basis") # Assuming 'basis' is the key for the basis in the .mat file
    file_path = r'/home/astar/Projects/data_from_matlab_code/mask_test.mat'  # Replace with your .mat file path
    mask_matlab = load_image_from_mat(file_path,"mask1") # Assuming 'mask' is the key for the mask in the .mat file
    file_path = r'/home/astar/Projects/matlab_code_v2/mem_data_1_x.mat'  # Replace with your .mat file path
    x_matlab = load_image_from_mat(file_path,"x") # Assuming 'x' is the key for the x coordinates in the .mat file
    file_path = r'/home/astar/Projects/matlab_code_v2/mem_data_1_y.mat'  # Replace with your .mat file path
    y_matlab = load_image_from_mat(file_path,"y") # Assuming 'y' is the key for the y coordinates in the .mat file
    file_path = r'/home/astar/Projects/data_from_matlab_code/mem_data_1_thetas.mat'  # Replace with your .mat file path
    thetas_matlab = load_image_from_mat(file_path, "thetas")
    # Convert the mask and image to PyTorch tensors
    img = img - np.mean(img)  # Normalize the image by subtracting the mean
    mask_tensor = torch.tensor(mask, dtype=torch.float32)  # Convert mask to tensor
    patch_tensor = torch.tensor(img, dtype=torch.float64)  # Convert image to tensor
    grid_size = 2  # Size of the grid for sampling
    stride = 4  # Stride for the sampling grid
    r = 20  # Radius of neighboring around grid point

    mask_tensor, row_idx, col_idx = get_sampling_grid(mask_tensor, grid_size, stride)
    row_idx, col_idx = select_points_within_boundary(img,r,row_idx,col_idx)
    # substitute the mask and grid coordinates with the one from MATLAB
    #mask_tensor = torch.tensor(mask_matlab)
    #mask_tensor = torch.tensor(mask_matlab)
    row_idx = torch.tensor(x_matlab)-1
    col_idx = torch.tensor(y_matlab)-1

    # Visualize the extracted patch and mask and the sampling grid
    #visualize_sampling_grid(mask_tensor, row_idx, col_idx)
    r_in = get_radius_of_inner_circle(r)  # Radius of the inner circle

    dataimg = patch_tensor.clone()  # Clone the original image to avoid modifying it
    # Initialize the output image and weight image
    imgout = torch.zeros_like(patch_tensor)  # Initialize the output image
    for iter in range(1):
        basis, thetas = get_basis(dataimg, mask_tensor, row_idx, col_idx, r, True)
        #compare_basis(basis,basis_matlab)
        #print("Thetas Pytorch - thetas Matlab", thetas - torch.tensor(thetas_matlab).squeeze())
        # Fit basis to data
        imgout = fit_basis_to_data(patch_tensor, basis, row_idx, col_idx, r)
        dataimg = imgout

        # Visualize subtracted image
        subtracted_img = patch_tensor - imgout*mask_tensor
        visualize_3_images(patch_tensor, imgout, subtracted_img,
                           title1="Original Patch",title2="Reconstructed Patch iter{}".format(iter),
                           title3="Subtracted Patch")
    #file_path = r'/home/astar/Projects/data_from_matlab_code/membrane_apprx_1iter.mat'  # Replace with your .mat file path
    #imgout_matlab = load_image_from_mat(file_path,
    #                                          "imgout")  # Assuming 'mask_dilate_before_loop' is the key for the dilated mask in the .mat file
    #compare_reconstr_image(imgout,imgout_matlab)
    plt.show()
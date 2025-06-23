from get_basis_test import visualize_3_images
import torch
from matplotlib import pyplot as plt
import time
from sampling_grid import get_sampling_grid
from membrane_subtract import membrane_subtract
from basis_fn import get_basis, get_basis_sequential, get_w_function
from testing_with_matlab import compare_reconstr_image, compare_masks
from testing_with_matlab import compare_angles
from fit_basis_to_data import fit_basis_to_data
from read_matlab import load_image_from_mat
import os

def set_mask_boundary_to_zero(mask,border=35):
    mask[:border]=0
    mask[-border:]=0
    mask[:,:border]=0
    mask[:,-border:]=0
    return mask
def convert_coords_from_matlab_to_torch(row_idx, col_idx):
    """
    Convert coordinates from MATLAB (1-based) to Python (0-based).

    Args:
        row_idx (torch.Tensor): Row indices from MATLAB.
        col_idx (torch.Tensor): Column indices from MATLAB.

    Returns:
        tuple: Converted row and column indices as torch tensors.
    """
    row_idx = row_idx - 1  # Convert to 0-based indexing
    col_idx = col_idx - 1  # Convert to 0-based indexing
    row_idx = torch.tensor(row_idx).squeeze().to(torch.int64)
    col_idx = torch.tensor(col_idx).squeeze().to(torch.int64)
    return row_idx, col_idx

def compare_basis(img,basis, basis_matlab,row_idx, col_idx,r,angles, angles_matlab):
    """
    Compare the basis functions from PyTorch and MATLAB.

    Args:
        basis (torch.Tensor): Basis functions from PyTorch.
        basis_matlab (torch.Tensor): Basis functions from MATLAB.
    """
    if basis.shape != basis_matlab.shape:
        raise ValueError(f"Shape mismatch: PyTorch {basis.shape}, MATLAB {basis_matlab.shape}")

    basis_matlab = torch.tensor(basis_matlab, dtype=torch.float64)
    norm_of_basis_diff = torch.linalg.matrix_norm(basis - basis_matlab, ord='fro', dim=(-2, -1))
    diff_idx = torch.argsort(norm_of_basis_diff)
    max_diff_idx = diff_idx[-1]  # Index of the maximum difference
    cur_patch = img.detach().clone()
    cur_patch[row_idx[max_diff_idx]-2:row_idx[max_diff_idx]+2, col_idx[max_diff_idx]-2:col_idx[max_diff_idx]+2] = 255
    print(f"Max difference is in basis:{max_diff_idx}, equal to {norm_of_basis_diff[max_diff_idx]}")
    visualize_3_images(basis[max_diff_idx], basis_matlab[max_diff_idx],basis[max_diff_idx] - basis_matlab[max_diff_idx], f"PyTorch basis {angles[max_diff_idx]}",f"MATLAB basis {angles_matlab[max_diff_idx]}", "Pytorch - matlab")
    #visualize_3_images(basis[max_diff_idx], basis_matlab[max_diff_idx],cur_patch, f"PyTorch basis {angles[max_diff_idx]}",f"MATLAB basis {angles_matlab[max_diff_idx]}", "Patch around grid point")


def subtrack_membrane_test():
    # Extract a small patch and its mask
    r = 20
    nb_iter = 1
    maindir = r"/home/astar/Projects/data_from_matlab_code"
    file_path = os.path.join(maindir,r'mk_1.mat')
    img, mask = load_image_from_mat(file_path, ["img","mask"])

    img = torch.tensor(img, dtype=torch.float64)
    mask_org= torch.tensor(mask, dtype=torch.float64)
    mask_org=set_mask_boundary_to_zero(mask_org)
    mask = mask_org.clone()  # Create a copy of the mask for processing
    img = img - torch.mean(img)  # Center the patch around zero
    file_path = os.path.join(maindir,r"mk_1_thetas.mat")
    angles_matlab = load_image_from_mat(file_path, "theta_array")
    mask, row_idx, col_idx = get_sampling_grid(mask, 4, 4)  # Get the sampling grid from the mask
    print(f"Number of grid points: {row_idx.shape[0]}")
    file_path = os.path.join(maindir, r"mk1_basis_iter1.mat")
    row_idx,col_idx,basis_matlab, mask_matlab, imgout_matlab = load_image_from_mat(file_path, ["x", "y", "basis", "mask1","imgout"])
    row_idx, col_idx = convert_coords_from_matlab_to_torch(row_idx, col_idx)
    print(f"Number of grid points in matlab: {row_idx.shape[0]}")
    compare_masks(mask, mask_matlab)
    dataimg = img.clone()
    for _ in range(3):
        basis,angles = get_basis(dataimg, mask, row_idx, col_idx, r,return_theta=True)
        #compare_basis(img,basis, basis_matlab,row_idx,col_idx,r,angles, angles_matlab)
        imgout_separate = fit_basis_to_data(img, basis, row_idx, col_idx, r)
        dataimg = imgout_separate
    compare_reconstr_image(imgout_separate, imgout_matlab)
    #compare_angles(angles,angles_matlab,row_idx,col_idx,basis)

    imgout_together, mask = membrane_subtract(img,mask_org,r, 1)
    visualize_3_images(imgout_separate,imgout_together,set_mask_boundary_to_zero(imgout_separate,40)-
                       set_mask_boundary_to_zero(imgout_together,40), "Imgout separate", "Imgout together", "Separate - Together")
    visualize_3_images(img,imgout_together,img - imgout_together*mask, "Org","Membrane", "Subtr")
    #file_path = r"/home/astar/Projects/data_from_matlab_code/mk_1_2.mat"
    #imgout_matlab = load_image_from_mat(file_path,"mem")
    #imgout_matlab = imgout_matlab[top:top+rows,left:left+cols]
    #imgout_matlab = torch.tensor(imgout_matlab)
    #visualize_3_images(imgout,imgout_matlab, imgout-imgout_matlab, "Pytorch membrane",
    #                   "Matlab membrane", "Pytorch-Matlab")

if __name__=="__main__":
    subtrack_membrane_test()
    plt.show()

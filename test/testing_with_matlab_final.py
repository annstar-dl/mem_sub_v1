from matplotlib import pyplot as plt
import os
import torch
from read_matlab import load_image_from_mat
from sampling_grid import get_sampling_grid, select_points_within_boundary
from basis_fn import get_basis
from visualizations import visualize_3_images
from fit_basis_to_data import fit_basis_to_data, fit_basis_to_data_batched
from membrane_subtract import membrane_subtract


def compare_reconstr_image(img, imgout, imgout_matlab,subtitle1, subtitle2, title="Reconstructed Image Comparison"):
    """
    Compare the reconstructed image from PyTorch with the one from MATLAB.

    Args:
        img (torch.Tensor): Original image tensor.
        imgout (torch.Tensor): Reconstructed image tensor from PyTorch.
        imgout_matlab (np.ndarray): Reconstructed image from MATLAB.
    """
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(title)
    imgout_matlab = torch.tensor(imgout_matlab, device="cpu", dtype=img.dtype)
    img = img.to("cpu")  # Ensure img is on the CPU for comparison
    imgout = imgout.to("cpu")  # Ensure imgout is on the CPU for comparison

    if torch.allclose(imgout, imgout_matlab, atol=1e-6):
        print("Reconstructed images are similar within the tolerance 1e-6.")

    diff = torch.abs(imgout - imgout_matlab)
    print(f"Max difference in reconstruction: {torch.max(diff).item()}")
    if torch.max(diff) > 1e-6:
        print("Reconstructed images are different, showing the difference image.")
    visualize_3_images(imgout, imgout_matlab, diff, subtitle1, subtitle2, "im1-im2", title)


def compare_angles(angles, angles_matlab):
    angles_matlab = torch.tensor(angles_matlab, device=angles.device, dtype=angles.dtype).squeeze()
    if angles.shape!=angles_matlab.shape:
        raise Exception(f"Shape tensors array from matlab and pytorch are not equal, "
                      f"and are matlab {angles_matlab.shape},"
                      f"and torch {angles.shape}")
    if not torch.allclose(angles, angles_matlab, atol=1e-6):
        angles_diff = torch.abs(angles - angles_matlab)
        angles_diff_binary = angles_diff > 1e-6
        nb_diff_angles = torch.sum(angles_diff_binary)
        print("Amount of angles that are different: ", nb_diff_angles.item())
        idx_max_diff = torch.argmax(angles_diff)
        print(f"Matlab angles are different from pytorch, example idx {idx_max_diff} pytorch {angles[idx_max_diff]} matlab {angles_matlab[idx_max_diff]} ")

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
    basis_matlab = torch.tensor(basis_matlab, device="cpu", dtype=basis.dtype)
    basis = basis.to("cpu")  # Ensure basis is on the CPU for comparison
    # Check if the values are similar
    diff_norm = torch.linalg.norm(basis - basis_matlab, dim=(1, 2))
    idx_max = torch.argmax(diff_norm)
    diff_binary = diff_norm > 1e-6
    nb_diff_basis = torch.sum(diff_binary)
    print("Amount of basis that are different: ", nb_diff_basis.item())
    print(f"Max difference in basis: {diff_norm[idx_max].item()} at index {idx_max.item()}")
    visualize_3_images(basis[idx_max], basis_matlab[idx_max], basis_matlab[idx_max] - basis[idx_max],
                       title1=f"Basis from PyTorch {idx_max}",
                       title2=f"Basis from MATLAB {idx_max}",
                       title3=f"Difference norm {diff_norm[idx_max]}")

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

def visualize_overlayed_sampling_grid(row_idx_py, row_idx, col_idx_py, col_idx, mask, r, title):
    """
    Visualize the sampling grid from PyTorch and MATLAB overlayed on the image.

    Args:
        row_idx_py (torch.Tensor): Row indices from PyTorch.
        row_idx (torch.Tensor): Row indices from MATLAB.
        col_idx_py (torch.Tensor): Column indices from PyTorch.
        col_idx (torch.Tensor): Column indices from MATLAB.
        img (torch.Tensor): Image tensor.
        r (int): Radius of neighborhood.
        title (str): Title for the plot.
    """
    plt.figure()
    plt.imshow(mask.numpy(), cmap='gray')
    plt.scatter(col_idx_py, row_idx_py, color='red', s=10, label = "Py coordinates")  # Plot the sampling grid points
    plt.scatter(col_idx, row_idx, color='green', s=10, label = "Matlab coordinates")  # Plot the sampling grid points
    plt.title("Sampling Grid on Mask")
    plt.legend()
    plt.axis('off')

def matlab_testing():
    r = 20
    step = 4
    nb_iter = 30
    maindir = r"/home/astar/Projects/data_from_matlab_code"
    file_path = os.path.join(maindir,r'mk_1.mat')
    img, mask = load_image_from_mat(file_path, ["img","mask"])
    img = torch.tensor(img, dtype=torch.float64)
    mask= torch.tensor(mask, dtype=torch.float64)
    mask_org = mask.clone()
    img = img - torch.mean(img)  # Center the patch around zero

    mask_py,row_idx_py, col_idx_py = get_sampling_grid(mask, 4, step)
    file_path = os.path.join(maindir, r'mk_1_coord_theta_basis.mat')
    row_idx, col_idx,angles_matlab, basis_matlab = load_image_from_mat(file_path, ["x", "y","theta_array", "basis"])
    row_idx, col_idx = convert_coords_from_matlab_to_torch(row_idx, col_idx)
    visualize_overlayed_sampling_grid(row_idx_py,row_idx, col_idx_py, col_idx, mask, r, "Sampling grid from PyTorch and MATLAB")


    basis,angles = get_basis(img, mask, row_idx, col_idx, r, return_theta=True)

    compare_basis(basis,basis_matlab)
    compare_angles(angles, angles_matlab)
    #compare fit bases
    #take basis and mask from matlab
    file_path = os.path.join(maindir, r"mk1_basis_iter1.mat")
    _, _, basis_matlab, mask_matlab, imgout_matlab = load_image_from_mat(file_path,
                                                                              ["x", "y", "basis", "mask1", "imgout"])
    basis = torch.tensor(basis_matlab, dtype=torch.float64)
    mask = torch.tensor(mask_matlab, dtype=torch.float64)
    #compare membranes
    imgout = fit_basis_to_data_batched(img, basis, row_idx, col_idx, r, 0.025, nb_iter, step)
    compare_reconstr_image(img, imgout, imgout_matlab,subtitle1="Pytorch",subtitle2="Matlab", title="Reconstructed Image from PyTorch vs MATLAB")

    #compare imgout on pytorch with mask from pytorch to substract_membrane function output
    imgout_together = membrane_subtract(img, mask_org)
    mask,row_idx, col_idx = mask_py, row_idx_py, col_idx_py
    row_idx, col_idx = select_points_within_boundary(img,r, row_idx, col_idx)
    dataimg = img.detach().clone()
    for i in range(3):
        basis = get_basis(dataimg, mask, row_idx, col_idx, r)
        imgout_separate = fit_basis_to_data_batched(img, basis, row_idx, col_idx, r, 0.025, nb_iter, step)
        dataimg = imgout_separate

    compare_reconstr_image(img, imgout_together, imgout_separate,subtitle1="Together",subtitle2="Separate",
                           title="Reconstructed Image from separate calc and membrane_subtract")




if __name__ == "__main__":
    # Example usage
    matlab_testing()
    plt.show()  # Show all plots at once
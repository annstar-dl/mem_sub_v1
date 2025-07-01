from align_image import rotate_images_kornia
import torch
from matplotlib import pyplot as plt
from basis_fn import get_basis, get_w_function, get_radius_of_inner_circle
from fit_basis_to_data import fit_basis_to_data
from read_matlab import load_image_from_mat
from visualizations import visualize_3_images
from testing_with_matlab_final import convert_coords_from_matlab_to_torch
import os

def set_mask_boundary_to_zero(mask,border=35):
    mask[:border]=0
    mask[-border:]=0
    mask[:,:border]=0
    mask[:,-border:]=0
    return mask

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
    mask_matlab = torch.tensor(mask_matlab, device=mask.device, dtype=mask.dtype)
    diff_mask = mask_matlab - mask
    visualize_3_images(mask, mask_matlab,diff_mask,"Mask from PyTorch", "Mask from MATLAB","Matlab - PyTorch")

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
        print("Amount of angles that are different: ", nb_diff_angles)
        idx_max_diff = torch.argmax(angles_diff)
        raise Exception(f"Matlab angles are different from pytorch, example idx {idx_max_diff} pytorch {angles[idx_max_diff]} matlab {angles_matlab[idx_max_diff]} ")

def investigate_wrong_angle():
    r = 20
    maindir = r"/home/astar/Projects/data_from_matlab_code"
    file_path = os.path.join(maindir,r'mk_1.mat')
    img, mask = load_image_from_mat(file_path, ["img","mask"])

    img = torch.tensor(img, dtype=torch.float64)
    mask = torch.tensor(mask, dtype=torch.float64)
    mask = set_mask_boundary_to_zero(mask, border=35)  # Set the boundary of the mask to zero
    img = img - torch.mean(img)  # Center the patch around zero
    file_path = os.path.join(maindir,r"mk_1_thetas.mat")
    angles_matlab = load_image_from_mat(file_path, "theta_array")
    file_path = os.path.join(maindir, r"mk1_basis_iter1.mat")
    row_idx,col_idx,basis_matlab, mask_matlab, imgout_matlab = load_image_from_mat(file_path, ["x", "y", "basis", "mask1","imgout"])
    row_idx, col_idx = convert_coords_from_matlab_to_torch(row_idx, col_idx)
    _,angles = get_basis(img, mask, row_idx, col_idx, r, return_theta=True)  # Get the basis functions and angles
    #compare_angles(angles,angles_matlab)
    wrong_idx = 7452
    row_idx = row_idx[wrong_idx]
    col_idx = col_idx[wrong_idx]
    angles = torch.range(-90, 90, 10, dtype=torch.float64)
    patch = img[row_idx-r:row_idx+r+1, col_idx-r:col_idx+r+1]
    patch = patch[None,None,:,:].expand(len(angles),-1,-1, -1)  # Expand the patch to match the number of angles
    rot_patch = rotate_images_kornia(patch, angles).squeeze()
    r_in = get_radius_of_inner_circle(r)
    cntr = r
    rot_patch = rot_patch[:,cntr - r_in:cntr+r_in+1, cntr-r_in:cntr+r_in+1]  # Remove the inner circle
    w = get_w_function(r_in)
    profile = torch.sum(rot_patch * w, dim=2)  # Sum over the rows to get the profile
    profile = profile.unsqueeze(-1).expand(-1,-1,rot_patch.shape[-1])  # Expand the profile to match the shape of the rotated patch
    loss = torch.sum((rot_patch - profile) ** 2, dim=(1, 2))  # Calculate the loss for each angle
    plt.figure()
    for i in range(2*(len(angles)//2)):
        rot_patch_prof = torch.concat((rot_patch[i], profile[i]), dim=-1)  # Concatenate the rotated patch and its profile
        print(f"Angle: {angles[i].item():.2f}, Loss: {loss[i].item()}")
        plt.subplot(len(angles)//2, 2, i+1)
        plt.imshow(rot_patch_prof.detach().numpy(), cmap='gray')
        plt.suptitle(f"Profile Loss for Patch at ({row_idx}, {col_idx})")
        plt.axis("off")
        plt.title(f"Angle: {angles[i].item():.2f}, Loss: {loss[i].item():}")

    # Visualize the rotated patches and their profiles






if __name__ == "__main__":
    investigate_wrong_angle()
    plt.show()  # Show the plots if any

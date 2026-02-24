from mem_sub.membrane_est.align_image import align_single_patch, align_multiple_patches, rotate_images_kornia, \
    calculate_mse_loss
from mem_sub.membrane_est.basis_fn import get_radius_of_inner_circle
from mem_sub.membrane_est.sub_utils import add_patches_to_image, get_patches_from_image_adv_indexing, creat_idx_batches_for_parl_sum, add_patches_to_image_batched
from matplotlib import pyplot as plt
from mem_sub.membrane_est.utils import read_parameters_from_yaml_file, read_img, save_im
from mem_sub.membrane_est.basis_fn import create_gaussian_disc
from  mem_sub.membrane_est.membrane_estimation import prepare_micrograph
import PIL.Image as Image

import numpy as np
import torch

def get_Bs(dataimg, row_idx, col_idx, r):
    """
    Get bases from the image data multiple patches at once. Basis is a membrane profile at a point.

    Args:
        img (torch.Tensor): Input image of shape (H,W), where N is the number of samples
                            and H, W are the height and width of the image.
        dataimg (torch.Tensor): The image from previous processing step of shape (H,W).
        row_idx (torch.Tensor): X coordinates of grid of shape (N), number of grid points.
        col_idx (torch.Tensor): Y coordinates of grid of shape (N), number of grid points.
        r (int): Radius of a neighbourhood.

    Returns:
        tuple: (torch.Tensor, torch.Tensor)
        1.torch.Tensor: Bases(Reconstructed images of neighbourhoods around grid point)
         of shape (N,r,r), where N is the number of bases and r is inner radius of neighbourhood
         around sampling grid point.
        2.torch.Tensor: The angles of the membrane profiles at each grid point of shape (N,).
    """
    # Check if the input image is 2D or 3D
    if dataimg.dim() != 2:
        raise ValueError("Input image must be a 2D tensor, got {} dimensions".format(dataimg.dim()))
    cntr = r
    r_in = get_radius_of_inner_circle(r)

    binaryImage, gaussWt = create_gaussian_disc(2*[(2*r_in+1)], r_in)  # Create a binary disc and Gaussian weights
    imgs_subset = get_patches_from_image_adv_indexing(dataimg, r, row_idx, col_idx)  # Get patches from the image using the specified radius
    imgs_subset = imgs_subset.unsqueeze(1)  # Add channel dimension
    # Move imgs_subset to GPU if available
    if torch.cuda.is_available():
        imgs_subset = imgs_subset.to("cuda")
        gaussWt = gaussWt.to("cuda")
    theta = align_multiple_patches_no_w(imgs_subset,cntr, r_in,-90.,90.0,1.0)  # Align the image using the center and radius
    basis = recon_mult_patches_no_w(imgs_subset, cntr, r_in, gaussWt, theta)  # Reconstruct the patch using the basis functions
    return basis, theta

def align_multiple_patches_no_w(imgs_subset, cntr, r, theta_b, theta_e, dtheta):
    """
    Align an image patch to the profile that matches the membrane crossection direction.
    :param img: torch.Tensor, Input image patch tensor of shape (C, H, W).
    :param cntr: int, Center of the image patch.
    :param r: int, Radius of a neighbourhood.
    :param w: torch.Tensor, Weights for the Gaussian kernel of shape (2*r+1, 2*r+1).
    :param theta_b: float, Starting angle for rotation in degrees.
    :param theta_e: float, Ending angle for rotation in degrees.
    :param dtheta: float, Step size for angle increment in degrees.
    :return:
    """
    theta_opt = align_multiple_patches_multires_no_w(imgs_subset, cntr, r,theta_b, theta_e, 10)
    # refine the angle by searching in a smaller range
    theta_opt = align_multiple_patches_multires_no_w(imgs_subset, cntr, r,theta_opt-10, theta_opt+10, dtheta)
    return theta_opt

def align_multiple_patches_multires_no_w(imgs_subset,cntr, r, theta_b, theta_e, dtheta):
    """
    Align multiple image patches to the profile of membranes. Every image patch has different angle rotation range.
    :param imgs_subset: torch.Tensor, Input image patches tensor of shape (N, C, H, W).
    :param cntr: int, center of the image patch.
    :param r: int, Radius of a neighbourhood.
    :param theta_b: float, Starting angle for rotation in degrees.
    :param theta_e: float, Ending angle for rotation in degrees.
    :param dtheta: float, Step size for angle increment in degrees.
    :return:
    """
    # Resize the angle to match the batch size
    # if range of angles is the same for all images then theta_b and theta_e are scalars
    if isinstance(theta_e, (int, float)) and isinstance(theta_e, (int, float)):
        angles = torch.arange(theta_b, theta_e+dtheta, dtheta)
        angles = angles.unsqueeze(-1).expand(-1,len(imgs_subset))  # N angles x M images
    # if range of angles is different for each image then theta_b and theta_e are lists
    elif isinstance(theta_b, (list, torch.Tensor)) and isinstance(theta_e, (list,torch.Tensor)):
        if len(theta_b) != len(theta_e):
            raise ValueError("theta_b and theta_e must have the same length if they are lists")
        angles = [torch.arange(b, e+dtheta, dtheta) for b, e in zip(theta_b, theta_e)]
        angles = torch.stack(angles,axis=0).transpose(0,1) # N angles x M images
    else:
        raise ValueError("theta_b and theta_e must be either both scalars or both lists")
    # check image dimensions
    if imgs_subset.dim() != 4:
        raise ValueError(f"Expected imgs to be a 4D tensor (N, C, H, W), got {imgs_subset.dim()} dimensions")
    # Initialize losses tensor
    losses = torch.zeros((len(angles),len(imgs_subset)), dtype=torch.float64, device=imgs_subset.device)
    for i in range(len(angles)):
        tmp_img = rotate_images_kornia(imgs_subset, angles[i])  # Rotate the images by the angles
        tmp_img = tmp_img[..., cntr - r:cntr + r + 1,cntr - r:cntr + r + 1]  # Crop the images to the neighbourhood size
        prof = torch.sum(tmp_img, dim=3)  # Calculate the profile for each rotated image
        prof = prof.unsqueeze(3).expand(-1, -1, -1, 2 * r + 1)  # Expand the profile into an image for each angle
        losses[i] = calculate_mse_loss(tmp_img, prof) # Calculate the MSE loss between the rotated images and the profile images
    loss_agr_min_idx = torch.argmin(losses, dim=0).to("cpu")  # Get the index of the minimum loss for each image
    best_angles = angles[loss_agr_min_idx,torch.arange(angles.size(1))]  # Get the best angles for each image
    return best_angles

def recon_mult_patches_no_w(imgs_subset, cntr, r_in, gaussWt, thetas):
    """
    Reconstruct the patch using the basis functions and the mask.

    Args:
        img1 (torch.Tensor): The image tensor of shape (C, H, W).
        cntr (int): Center of the patch.
        r_in (int): Radius of the inner circle.
        gaussWt (torch.Tensor): Gaussian weights for the patch.
        thetas (list(float)): Angle to rotate the image.

    Returns:
        torch.Tensor: The reconstructed patch tensor.
    """
    tmp = rotate_images_kornia(imgs_subset, thetas)  # Rotate the image by the angle
    tmp = tmp[...,cntr - r_in:cntr + r_in+1, cntr - r_in:cntr + r_in+1]  # Crop the image to the inner neighborhood size
    prof = tmp.sum(dim=3)  # Sum the profile across the columns
    prof = prof.unsqueeze(-1).expand(-1,-1,-1, 2 * r_in+1)  # Expand the profile into an image for each angle
    neg_thetas = -thetas
    prof = rotate_images_kornia(prof, neg_thetas)  # Rotate the profile back to the original orientation
    gaussWt = gaussWt.unsqueeze(0).unsqueeze(0)  # Ensure gaussWt is a 4D tensor for broadcasting
    reconstructed_patchs = prof * gaussWt  # Scale the profile by the Gaussian weights
    return reconstructed_patchs.squeeze(1)

def get_membrane(dataimg, row_idx, col_idx, r, step):
    """Get the membrane profile at each grid point.
    Args:
        dataimg (torch.Tensor): The image from previous processing step of shape (H,W).
        row_idx (torch.Tensor): X coordinates of grid of shape (N), number of grid points.
        col_idx (torch.Tensor): Y coordinates of grid of shape (N), number of grid points.
        r (int): Radius of a neighbourhood."""
    r_in = get_radius_of_inner_circle(r)  # Get the radius of the inner circle
    basis, theta = get_Bs(dataimg, row_idx, col_idx, r)  # Get the bases and angles for the membrane profile
    batched_row_idxs, batched_col_idxs, bases_idxs = creat_idx_batches_for_parl_sum(row_idx, col_idx, r_in, step)
    ones = torch.ones_like(dataimg)  # Create a tensor of ones with the same shape as the basis
    domains = get_patches_from_image_adv_indexing(ones, r_in, row_idx, col_idx)
    disk = create_disc(r_in)  # Create a disc-shaped weighting function for the membrane profile
    empty_img = torch.zeros_like(dataimg)
    if torch.cuda.is_available():
        disk = disk.to("cuda")
        domains = domains.to("cuda")
        empty_img = empty_img.to("cuda")
    domains = domains * disk.unsqueeze(0)
    basis = basis * disk.unsqueeze(0)  # Apply the disc-shaped weighting function to the basis


    imgout = add_patches_to_image_batched(basis, empty_img, r_in, batched_row_idxs, batched_col_idxs,
                                          bases_idxs)
    domains_sum = add_patches_to_image_batched(domains,empty_img, r_in, batched_row_idxs, batched_col_idxs,
                                          bases_idxs)
    imgout/=domains_sum  # Normalize the output image by the sum of the domains to get the average membrane profile
    return imgout

def create_disc(r):
    """Create a disc-shaped weighting function for the membrane profile.
    Args:
        r (int): Radius of a neighbourhood.
        r_in (int): Radius of the inner circle."""
    w = torch.zeros((2*r+1, 2*r+1))  # Initialize a tensor of zeros with the shape of the disc
    for i in range(-r,r+1):
        for j in range(-r,r+1):
            if i**2 + j**2 <= r**2:  # Check if the point is within the disc radius
                w[i+r, j+r] = 1.0  # Set the weight to 1 if it is within the disc radius
    return w

if __name__ == "__main__":
    fpath_img = r"/home/astar/Projects/vesicles_data/labeled_data/2025_sep+dec_separated_membrane/images/kcnq1(mackinnon)/008137117410956406710_VSM-71-3_345_005_Nov03_10.45.24_X+1Y-1-1_patch_aligned_doseweighted.jpg"

    fpath_mask = r"/home/astar/Projects/vesicles_data/labeled_data/2025_sep+dec_separated_membrane/labels/kcnq1(mackinnon)/008137117410956406710_VSM-71-3_345_005_Nov03_10.45.24_X+1Y-1-1_patch_aligned_doseweighted.png"

    img = read_img(fpath_img)  # Read the input image
    mask = read_img(fpath_mask, mask=True)  # Read the segmentation mask
    parameters = read_parameters_from_yaml_file()  # Read parameters from the YAML file
    img_tensor, mask_blured, row_indices, col_indices = prepare_micrograph(img, mask, parameters, parameters["r"])
    membrane = get_membrane(img_tensor, row_indices, col_indices, 20,4)
    membrane = membrane.to("cpu").numpy()  # Move the membrane tensor to CPU and convert to NumPy array for visualization
    plt.figure()
    plt.imshow(membrane, cmap='gray')
    plt.show()
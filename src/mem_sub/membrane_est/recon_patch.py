import torch
from mem_sub.membrane_est.align_image import rotate_images_kornia

def recon_patch(img1, cntr, r_in, w, gaussWt, theta):
    """
    Reconstruct the patch using the basis functions and the mask.

    Args:
        img1 (torch.Tensor): The image tensor of shape (C, H, W).
        cntr (int): Center of the patch.
        r_in (int): Radius of the inner circle.
        w (torch.Tensor): Weights for the Gaussian kernel.
        gaussWt (torch.Tensor): Gaussian weights for the patch.
        theta (float): Angle to rotate the image.

    Returns:
        torch.Tensor: The reconstructed patch tensor.
    """
    tmp = rotate_images_kornia(img1.unsqueeze(0).unsqueeze(0), theta)  # Rotate the image by the angle
    tmp = tmp.squeeze()  # Remove all dimensions of size 1
    tmp = tmp[cntr - r_in:cntr + r_in+1, cntr - r_in:cntr + r_in+1]  # Crop the image to the inner neighborhood size
    prof = tmp * w  # Apply the weighted average function to the rotated image
    prof = prof.sum(dim=1)  # Sum the profile across the columns
    prof = prof.unsqueeze(1).expand(-1, 2 * r_in+1)  # Expand the profile into an image for each angle
    prof = rotate_images_kornia(prof.unsqueeze(0).unsqueeze(0), -theta)  # Rotate the profile back to the original orientation
    prof = prof.squeeze()  # Remove all dimensions of size 1
    reconstructed_patch = prof * gaussWt  # Scale the profile by the Gaussian weights
    return reconstructed_patch

def recon_mult_patches(imgs_subset, cntr, r_in, w, gaussWt, thetas):
    """
    Reconstruct the patch using the basis functions and the mask.

    Args:
        img1 (torch.Tensor): The image tensor of shape (C, H, W).
        cntr (int): Center of the patch.
        r_in (int): Radius of the inner circle.
        w (torch.Tensor): Weights for the Gaussian kernel.
        gaussWt (torch.Tensor): Gaussian weights for the patch.
        thetas (list(float)): Angle to rotate the image.

    Returns:
        torch.Tensor: The reconstructed patch tensor.
    """
    tmp = rotate_images_kornia(imgs_subset, thetas)  # Rotate the image by the angle
    tmp = tmp[...,cntr - r_in:cntr + r_in+1, cntr - r_in:cntr + r_in+1]  # Crop the image to the inner neighborhood size
    w = w.unsqueeze(0).unsqueeze(0)  # Ensure w is a 4D tensor for broadcasting
    prof = tmp * w  # Apply the Gaussian profile weights to the rotated image
    prof = prof.sum(dim=3)  # Sum the profile across the columns
    prof = prof.unsqueeze(-1).expand(-1,-1,-1, 2 * r_in+1)  # Expand the profile into an image for each angle
    neg_thetas = -thetas
    prof = rotate_images_kornia(prof, neg_thetas)  # Rotate the profile back to the original orientation
    gaussWt = gaussWt.unsqueeze(0).unsqueeze(0)  # Ensure gaussWt is a 4D tensor for broadcasting
    reconstructed_patchs = prof * gaussWt  # Scale the profile by the Gaussian weights
    return reconstructed_patchs.squeeze(1)
import torch
from align_image import rotate_images_kornia

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
    tmp = tmp[cntr - r_in:cntr + r_in+1, cntr - r_in:cntr + r_in+1]  # Crop the image to the neighborhood size
    prof = tmp * w  # Apply the Gaussian profile weights to the rotated image
    prof = prof.sum(dim=1)  # Sum the profile across the columns
    prof = prof.unsqueeze(1).expand(-1, 2 * r_in+1)  # Expand the profile into an image for each angle
    prof = rotate_images_kornia(prof.unsqueeze(0).unsqueeze(0), -theta)  # Rotate the profile back to the original orientation
    prof = prof.squeeze()  # Remove all dimensions of size 1
    reconstructed_patch = prof * gaussWt  # Scale the profile by the Gaussian weights
    return reconstructed_patch
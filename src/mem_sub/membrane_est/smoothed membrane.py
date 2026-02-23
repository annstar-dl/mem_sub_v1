

def get_Bs():
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
    theta = align_multiple_patches(imgs_subset,cntr, r_in,w,-90.,90.0,1.0)  # Align the image using the center and radius
    basis = recon_mult_patches(imgs_subset, cntr, r_in, w, gaussWt, theta)  # Reconstruct the patch using the basis functions
    return basis, theta

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
    prof = prof.sum(dim=3)  # Sum the profile across the columns
    prof = prof.unsqueeze(-1).expand(-1,-1,-1, 2 * r_in+1)  # Expand the profile into an image for each angle
    neg_thetas = -thetas
    prof = rotate_images_kornia(prof, neg_thetas)  # Rotate the profile back to the original orientation
    gaussWt = gaussWt.unsqueeze(0).unsqueeze(0)  # Ensure gaussWt is a 4D tensor for broadcasting
    reconstructed_patchs = prof * 1 #gaussWt  # Scale the profile by the Gaussian weights
    return reconstructed_patchs.squeeze(1)
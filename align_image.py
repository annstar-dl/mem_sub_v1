import torch
import kornia
import numpy as np
import matplotlib.pyplot as plt


def calculate_mse_loss(original, rotated):
    """Calculate the Mean Squared Error (MSE) loss between original and rotated images."""
    return torch.sum((original - rotated) ** 2, dim=(1, 2, 3))  # MSE across batch, height, and width

def rotate_images_kornia(images, angles):
    """    Rotate a batch of images using Kornia.
    Args:
        images (torch.Tensor): Batch of images with shape (B, C, H, W).
        angle_degrees (float or list): Angle in degrees to rotate the images, or list of angles for each image.

        If you are using kornia in your research-related documents, it is recommended that you cite the paper.
        @inproceedings{eriba2019kornia,
          author    = {E. Riba, D. Mishkin, D. Ponsa, E. Rublee and G. Bradski},
          title     = {Kornia: an Open Source Differentiable Computer Vision Library for PyTorch},
          booktitle = {Winter Conference on Applications of Computer Vision},
          year      = {2020},
          url       = {https://arxiv.org/pdf/1910.02190.pdf}
    }
    """
    if images.dim() != 4:
        raise ValueError(f"Expected images to be a 4D tensor (B, C, H, W), got {images.dim()} dimensions")
    b = images.size(0) # Get the batch size
    # Resize the angle to match the batch size
    if isinstance(angles, (int, float)):
        angles = torch.tensor([angles] * b, dtype=torch.double)  # Repeat the angle for each image in the batch
    elif isinstance(angles, list):
        angles = torch.tensor(angles, dtype=torch.double)  # Convert list to tensor
        if len(angles) != b:
            raise ValueError(f"Expected angle_degrees to be a single value or a list of length {b}, got {len(angles)}")
    # conver angles to double precision
    angles = angles.to(torch.double)  # Ensure angles are in double precision
    # move angles to the same device as imgs_subset
    if images.device != angles.device:
        angles = angles.to(images.device)

    center = torch.tensor([[images.shape[-2] // 2, images.shape[-1] // 2]], dtype=torch.double, device=images.device)
    rotated_images = kornia.geometry.rotate(images,angles,center,mode="bilinear")
    return rotated_images


def align_multiple_patches(imgs_subset, cntr, r, w, theta_b, theta_e, dtheta):
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
    theta_opt = align_multiple_patches_multires(imgs_subset, cntr, r, w, theta_b, theta_e, 10)
    theta_opt = align_multiple_patches_multires(imgs_subset, cntr, r, w, theta_opt-10, theta_opt+10, dtheta)
    return theta_opt


def align_multiple_patches_multires(imgs_subset,cntr, r, w, theta_b, theta_e, dtheta):
    """
    Align multiple image patches to the profile that matches the membrane crossection direction.
    :param imgs: torch.Tensor, Input image patches tensor of shape (N, C, H, W).
    :param cntrs: list of int, Centers of the image patches.
    :param r: int, Radius of a neighbourhood.
    :param w: torch.Tensor, Weights for the Gaussian kernel of shape (2*r+1, 2*r+1).
    :param theta_b: float, Starting angle for rotation in degrees.
    :param theta_e: float, Ending angle for rotation in degrees.
    :param dtheta: float, Step size for angle increment in degrees.
    :return:
    """
    if isinstance(theta_e, (int, float)) and isinstance(theta_e, (int, float)):
        angles = torch.arange(theta_b, theta_e+dtheta, dtheta)
        angles = angles.unsqueeze(-1).expand(-1,len(imgs_subset))  # N angles x M images
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
    losses = torch.zeros((len(angles),len(imgs_subset)), dtype=torch.float64, device=imgs_subset.device)  # Initialize losses tensor
    for i in range(len(angles)):
        tmp_img = rotate_images_kornia(imgs_subset, angles[i])  # Rotate the images by the angles
        tmp_img = tmp_img[..., cntr - r:cntr + r + 1,cntr - r:cntr + r + 1]  # Crop the images to the neighbourhood size
        w_exp = w.unsqueeze(0).unsqueeze(0)
        prof = tmp_img * w_exp  # Apply the Gaussian weights to the rotated images
        # visualize_all_rotations(prof,supertitle="Weighted Rotated Images")
        prof = torch.sum(prof, dim=3)  # Calculate the profile for each rotated image
        prof = prof.unsqueeze(3).expand(-1, -1, -1, 2 * r + 1)  # Expand the profile into an image for each angle
        losses[i] = calculate_mse_loss(tmp_img, prof) # Calculate the MSE loss between the rotated images and the original images
    loss_agr_min_idx = torch.argmin(losses, dim=0).to("cpu")  # Get the index of the minimum loss for each image
    best_angles = angles[loss_agr_min_idx,torch.arange(angles.size(1))]  # Get the best angles for each image
    return best_angles

def align_single_patch(img, cntr, r, w, theta_b, theta_e, dtheta):
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
    theta_opt = align_single_patch_multires(img, cntr, r, w, theta_b, theta_e, 10)
    theta_opt = align_single_patch_multires(img, cntr, r, w, theta_opt-10, theta_opt + 10, dtheta)
    return theta_opt


def align_single_patch_multires(img, cntr, r, w, theta_b, theta_e, dtheta):
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

    angles_list = np.arange(theta_b, theta_e+dtheta, dtheta).tolist()
    # check image dimensions
    if img.dim() < 3:
        img = img.unsqueeze(0)  # Ensure img is a 4D tensor (C, H, W)
        if img.dim() < 4:
            img = img.unsqueeze(0)
    tmp_img = img.expand(len(angles_list), -1, -1, -1).contiguous()  # Efficiently repeat the image for each angle in the batch

    tmp_img = rotate_images_kornia(tmp_img, angles_list)  # Rotate the images by the angles
    tmp_img = tmp_img[..., cntr - r:cntr + r+1, cntr - r:cntr + r+1]  # Crop the images to the neighbourhood size
    w_exp = w.unsqueeze(0).unsqueeze(0)
    prof = tmp_img * w_exp  # Apply the Gaussian weights to the rotated images
    #visualize_all_rotations(prof,supertitle="Weighted Rotated Images")
    prof = torch.sum(prof, dim=3) # Calculate the profile for each rotated image
    prof = prof.unsqueeze(3).expand(-1, -1, -1, 2 * r+1)  # Expand the profile into an image for each angle

    #visualize_all_rotations(tmp_img,supertitle="Rotated Images")
    loss = calculate_mse_loss(tmp_img, prof)  # Calculate the MSE loss between the rotated images and the profile
    #visualize_all_rotations(prof,loss, "Profiles")
    loss_agr_min_idx = torch.argmin(loss,dim=0)
    return angles_list[loss_agr_min_idx]  # Get the best angle

def visualize_all_rotations(imgs, loss=None,supertitle=None):
    """
    Visualize all rotations of the image patch.
    :param img: torch.Tensor, Input image patch tensor of shape (C, H, W).
    :param cntr: int, Center of the image patch.
    :param r: int, Radius of the neighbourhood.
    :param w: torch.Tensor, Weights for the Gaussian kernel of shape (2*r+1, 2*r+1).
    :param theta_b: float, Starting angle for rotation in degrees.
    :param theta_e: float, Ending angle for rotation in degrees.
    :param dtheta: float, Step size for angle increment in degrees.
    """
    plt.figure()
    fig, axes = plt.subplots(1, len(imgs), figsize=(15, 5))
    for i, img in enumerate(imgs):
        axes[i].imshow(img[0].cpu().numpy(),cmap="gray")  # Convert tensor to numpy array for visualization
        axes[i].axis('off')
        if loss is not None:
            axes[i].set_title(f'Loss: {loss[i]:.2f}, Rot {i+1}')
        else:
            axes[i].set_title(f'Rotation {i+1}')
    if not supertitle is None:
        plt.suptitle(supertitle, fontsize=16)


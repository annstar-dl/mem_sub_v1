import torch
import kornia
import numpy as np
import matplotlib.pyplot as plt

def align_image_all(img, cntr, r,w,theta_b, theta_e, dtheta):
    """
    Align the image.

    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W).
        cntr (int): Center of the image.
        r (int): Radius of the neirghbourhood.
        w (torch.Tensor): Weights for the Gaussian kernel.
        theta_b (float): Starting angle for rotation.
        theta_e (float): Ending angle for rotation.
        dtheta (float): Step size for angle increment.
    Returns:
        torch.Tensor: angle of aligned image.
    """
    theta_opt = align_singe_patch(img, cntr, r, w, theta_b, theta_e, dtheta)  # Align the image patch
    #theta_opt = align_image_multires(img, cntr, r, w, theta_b, theta_e, 1)
    #theta_opt = align_image_multires(img, cntr, r, w, theta_opt-10, theta_opt + 10, dtheta)
    return theta_opt

def calculate_mse_loss(original, rotated):
    """Calculate the Mean Squared Error (MSE) loss between original and rotated images."""
    return torch.mean((original - rotated) ** 2, dim=(1, 2, 3))  # MSE across batch, height, and width

def rotate_images_kornia(images, angle):
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
    if isinstance(angle, (int, float)):
        angle = torch.tensor([angle] * b,dtype=torch.float32)  # Repeat the angle for each image in the batch
    elif isinstance(angle, list):
        angle = torch.tensor(angle, dtype=torch.float32)  # Convert list to tensor
        if len(angle) != b:
            raise ValueError(f"Expected angle_degrees to be a single value or a list of length {b}, got {len(angle)}")
    # define the rotation center
    center = torch.ones(b, 2)
    center[...,0] = images.size(2) / 2
    center[...,1] = images.size(3) / 2
    # Rotate the batch
    rotated_images = kornia.geometry.rotate(images, angle, center)
    return rotated_images

def align_single_patch(img, cntr, r, w, theta_b, theta_e, dtheta):
    """
    Align an image patch to the profile that matches the membrane crossection direction.
    :param img: torch.Tensor, Input image patch tensor of shape (C, H, W).
    :param cntr: int, Center of the image patch.
    :param r: int, Radius of the neighbourhood.
    :param w: torch.Tensor, Weights for the Gaussian kernel of shape (2*r+1, 2*r+1).
    :param theta_b: float, Starting angle for rotation in degrees.
    :param theta_e: float, Ending angle for rotation in degrees.
    :param dtheta: float, Step size for angle increment in degrees.
    :return:
    """

    angles_list = np.arange(theta_b, theta_e, dtheta).tolist()
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


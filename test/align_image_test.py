import torch
from extract_small_patch import extract_small_patch_with_mask
from  sampling_grid import get_sampling_grid, select_points_within_boundary
from align_image import rotate_images_kornia, align_single_patch
from basis_fn import get_w_function, get_radius_of_inner_circle
import matplotlib.pyplot as plt
from sampling_grid_test import visualize_sampling_grid


def visualize_im(im, title="Image", add_figure=True):
    """
    Visualize the image tensor.
    Args:
        im (torch.Tensor): Image tensor of shape (C, H, W).
    """
    if add_figure:
        plt.figure()
    if im.dim()>2:
        im = im.permute(1, 2, 0)  # Change shape to (H, W, C) for visualization
        plt.imshow(im)
    else:
        plt.imshow(im, cmap='gray')  # For single channel images
    plt.axis('off')
    plt.title(title)

def visualize_2_images(im1, im2, title1="Image 1", title2="Image 2"):
    """
    Visualize two images side by side.
    Args:
        im1 (torch.Tensor): First image tensor of shape (C, H, W).
        im2 (torch.Tensor): Second image tensor of shape (C, H, W).
        title1 (str): Title for the first image.
        title2 (str): Title for the second image.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    visualize_im(im1, title=title1, add_figure=False)
    plt.subplot(1, 2, 2)
    visualize_im(im2, title=title2, add_figure=False)


def visualize_3_images(im1, im2, im3, title1="Image 1", title2="Image 2", title3="Image 3"):
    """
    Visualize three images side by side.
    Args:
        im1 (torch.Tensor): First image tensor of shape (C, H, W).
        im2 (torch.Tensor): Second image tensor of shape (C, H, W).
        im3 (torch.Tensor): Third image tensor of shape (C, H, W).
        title1 (str): Title for the first image.
        title2 (str): Title for the second image.
        title3 (str): Title for the third image.
    """
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    visualize_im(im1, title=title1, add_figure=False)
    plt.subplot(1, 3, 2)
    visualize_im(im2, title=title2, add_figure=False)
    plt.subplot(1, 3, 3)
    visualize_im(im3, title=title3, add_figure=False)


def extract_small_patch_with_padding(img, x, y, r):
    """
    Extract a small patch from the image. Image is padded in case the patch goes out of bounds.

    Args:
        img (torch.Tensor): The image tensor of shape (C, H, W).
        x (int): X coordinate of the center of the patch.
        y (int): Y coordinate of the center of the patch.
        r (int): Radius of the patch to extract.

    Returns:
        torch.Tensor: The extracted patch tensor of shape (C, 2*r+1, 2*r+1).
    """
    img = img.unsqueeze(0) if img.dim() == 3 else img  # Ensure img is 4D tensor (C, H, W)
    img_padded = torch.nn.functional.pad(img, (r, r, r, r), mode='constant', value=0)  # Pad the image
    # Extract the patch from the padded image
    patch = img_padded[:,:, y:y+2*r, x:x+2*r]  # Extract the patch
    return patch.squeeze(0)


def align_image_test():
    """
    Align the image based on the mask and sampling grid.

    Args:
        img (torch.Tensor): The image tensor of shape (C, H, W).
        mask (torch.Tensor): The mask tensor of shape (H, W).
        grid_size (int): Size of the grid for sampling.
        stride (int): Stride for the sampling grid.

    Returns:
        torch.Tensor: The aligned image tensor.
    """

    grid_size = 2  # Size of the grid
    stride = 4  # Stride for the sampling grid
    r = 14 # Radius of neighboring around grid point
    cntr = r+1  # Center of the neighborhood
    r_in = get_radius_of_inner_circle(r)  # Radius of the inner circle
    #get Gaussian weights for the neighborhood
    w = get_w_function(r_in)
    # Visualize Gaussian weights
    visualize_im(w, title="Gaussian Weights")
    # Align the image patch using the mask and grid
    cur_patch_tensor = get_w_function(r).unsqueeze(0)
    angle = align_single_patch(cur_patch_tensor, cntr , r_in, w, -90.0, 90.0, 1.0)
    # Rotate the patch in the opposite direction
    rot_patch_tensor = rotate_images_kornia(cur_patch_tensor.unsqueeze(0), angle).squeeze(0).squeeze(0)  # Rotate the patch
    rot_patch_tensor = rot_patch_tensor[...,cntr-r_in:cntr+r_in+1, cntr-r_in:cntr+r_in+1]  # Crop the rotated patch to the original size
    rot_patch_tensor_scale = rot_patch_tensor * w  # Scale the rotated patch by the weights
    rot_patch_profile = rot_patch_tensor_scale.sum(dim=(1))
    rot_patch_profile = rot_patch_profile.unsqueeze(1).repeat(1,2*r_in)  # Sum across height and width to get the profile
    # Visualize the original and rotated patches
    visualize_3_images(cur_patch_tensor[0],rot_patch_tensor, rot_patch_profile,title1=f"Original Patch",
                      title2=f"Rotated Patch (Angle: {angle:.2f} degrees)",
                      title3=f"Profile of Rotated Patch ")

if __name__== "__main__":
    align_image_test()
    plt.show()
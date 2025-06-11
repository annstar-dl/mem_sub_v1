import torch
from PIL import Image
import os
from sampling_grid import get_sampling_grid
from basis_fn import get_bases


def extract_small_patch():
    """
    Extract a small patch from a larger image.
    This function is a placeholder and should be implemented based on specific requirements.
    """
    # Example implementation: Load an image and extract a small patch
    data_dir = r"/home/astar/Projects/vesicles_data"
    image_name = r"slot6_100_0002ms"
    image_path = os.path.join(data_dir, "test", image_name + ".jpg")
    image = Image.open(image_path).convert("RGB")

    # Define the patch size and position
    patch_size = (160, 160)  # Width, Height
    left = 1000  # X coordinate of the top-left corner
    top = 1000  # Y coordinate of the top-left corner

    # Extract the patch
    patch = image.crop((left, top, left + patch_size[0], top + patch_size[1]))
    patch.show("Original Patch")
    # Convert the patch to a tensor
    patch_tensor = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1)
    return patch_tensor

def  membrane_subtract(img, mask):
    """    Subtract the membrane mask from the patch.
    Args:
        img (torch.Tensor): The micrograph(image with membranes) image tensor of shape (C, H, W).
        mask (torch.Tensor): The micrograph mask tensor of shape (H, W).
    Returns:
        torch.Tensor: The patch with the membrane subtracted.
    """
    img = img - torch.mean(img)
    mask, x, y = get_sampling_grid(mask, 2, 4)  # Get the sampling grid from the mask

    r = 14 # Radius of neighboring around grid point
    dataimg = img.clone()  # Clone the original image to avoid modifying it

    for _ in range(3):
        basis = get_bases(img,dataimg, mask, x, y, r)


    return None
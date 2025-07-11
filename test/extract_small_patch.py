from PIL import Image
import os
import numpy as np
import torch

def extract_small_patch_with_mask(left=200, top=300, patch_size=(160, 160)):
    """
    Extract a small patch from a larger image.
    This function is a placeholder and should be implemented based on specific requirements.
    Args:
        left (int): X coordinate of the top-left corner of the patch.
        top (int): Y coordinate of the top-left corner of the patch.
        patch_size (tuple) or None: Size of the patch as (width, height). IF None, the patch will be extracted to the right and down from the specified coordinates.
    """
    # Example implementation: Load an image and extract a small patch
    data_dir = r"/home/astar/Projects/vesicles_data/data_analyses"
    image_name = r"slot6_100_0002ms"
    image_path = os.path.join(data_dir, "test", image_name + ".jpg")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} does not exist.")
    image = Image.open(image_path).convert("RGB")
    image = image.convert(mode="L")  # Convert to grayscale if needed
    mask_path = os.path.join(data_dir, "labels", image_name + ".png")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} does not exist.")
    mask = Image.open(mask_path).convert("L")  # Load the mask as grayscale
    # Extract the patch
    if patch_size is None:
        patch_size = (image.width - left, image.height - top)
    patch = image.crop((left, top, left + patch_size[0], top + patch_size[1]))
    # Extract the corresponding mask patch
    mask_patch = mask.crop((left, top, left + patch_size[0], top + patch_size[1]))
    # Convert the patch to a tensor
    patch_tensor = torch.tensor(np.array(patch), dtype=torch.float64).unsqueeze(0)
    mask_tensor = torch.tensor(np.array(mask_patch), dtype=torch.float64)
    return patch_tensor, mask_tensor
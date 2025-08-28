import copy

from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import importlib.util
import sys

file_path = r'/bg_estimation.py'
module_name = 'bg_estimation'

spec = importlib.util.spec_from_file_location(module_name, file_path)
bg_estimation = importlib.util.module_from_spec(spec)
sys.modules[module_name] = bg_estimation
spec.loader.exec_module(bg_estimation)

# Now you can use module.your_function()

def read_mask(mask_path):
    """
    Read a mask from a file and convert it to a tensor.

    Args:
        mask_path (str): Path to the mask file.

    Returns:
        torch.Tensor: Mask tensor of shape (H, W).
    """
    mask = Image.open(mask_path).convert("F")  # Convert to grayscale
    return np.array(mask)

def get_background(img,mask):
    """Flatten(even out) the background of the image using Gaussian smoothing.
    Args:
        img (numpy.ndarray): Input image array.
        mask (numpy.ndarray): Input mask array.
    Returns:
        numpy.ndarray: Image with flattened background.
        """
    img = copy.deepcopy(img)
    img_background, diff = bg_estimation.get_background(img, mask, sigma=30.0)
    return img_background, diff

def read_image(image_path):
    """
    Read an image from a file and convert it to a tensor.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Image tensor of shape (H, W).
    """
    img = Image.open(image_path)
    img = np.array(img,dtype=np.float32)  # Convert to numpy array
    return img
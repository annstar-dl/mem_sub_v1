import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

def get_background(img: np.ndarray, mask: np.ndarray, sigma:
                   float = 30.0) -> (np.ndarray,np.ndarray):
    """
    Get the background of an image by applying Gaussian smoothing and masking.

    Args:
        img (np.ndarray): Input image array, type float64.
        mask (np.ndarray): Mask array of membrane
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        np.ndarray: Background image with the same shape as the input image.
    """
    # Ensure mask is in float64 format for compatibility with Gaussian filter
    # Invert the mask to create a background mask
    if not mask.dtype == np.float64:
        mask = mask.astype(np.float64)
    if not img.dtype == np.float64:
        img = img.astype(np.float64)
    bg_mask = 1 - mask
    #Smooth image and mask
    img_smoothed = cv2.GaussianBlur(img*bg_mask, (0, 0), sigmaX=sigma)
    bg_mask_smoothed = cv2.GaussianBlur(bg_mask, (0, 0), sigmaX=sigma)

    # estimate background by dividing smoothed image by smoothed mask
    img_background = img_smoothed / (bg_mask_smoothed + 1e-6)  # Avoid division by zero
    diff = np.subtract(img, img_background)
    return img_background, diff
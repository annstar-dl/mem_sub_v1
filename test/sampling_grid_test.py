import numpy as np
import torch
from  sampling_grid import get_sampling_grid
import os
from PIL import Image
import matplotlib.pyplot as plt
from extract_small_patch import extract_small_patch_with_mask

def visualize_patch(patch_tensor, mask_tensor, subtitle="Patch and Mask"):
    """
    Visualize the extracted patch and its corresponding mask.
    """

    # Convert tensors to numpy arrays for visualization
    patch_np = patch_tensor.permute(1, 2, 0).numpy()  # Change to (H, W, C)
    mask_np = mask_tensor.squeeze().numpy()  # Remove channel dimension

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Patch")
    plt.imshow(patch_np.astype(np.uint8))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask_np, cmap='gray')
    plt.axis('off')
    plt.suptitle(subtitle)

def visualize_sampling_grid(mask_tensor, y, x):
    """
    Visualize the sampling grid on the mask.
    """
    plt.figure()
    plt.imshow(mask_tensor.numpy(), cmap='gray')
    plt.scatter(x, y, color='red', s=10)  # Plot the sampling grid points
    plt.title("Sampling Grid on Mask")
    plt.axis('off')

def get_sampling_grid_test(grid_size, stride):
    """
    Test function to get the sampling grid from a mask.
    """
    # Define the patch size and position
    patch_size = (160, 160)  # Width, Height
    left = 200  # X coordinate of the top-left corner
    top = 300  # Y coordinate of the top-left corner
    patch_tensor, mask_tensor = extract_small_patch_with_mask(left, top, patch_size)
    visualize_patch(patch_tensor, mask_tensor,"Original Patch and Mask")
    mask_tensor, row_idx, col_idx = get_sampling_grid(mask_tensor, grid_size, stride)
    visualize_patch(patch_tensor, mask_tensor,"Dilated Patch and Mask")
    return mask_tensor, row_idx, col_idx

if __name__ == "__main__":
    # Example usage
    grid_size = 2  # Size of the grid
    stride = 4  # Stride for the sampling grid
    sampling_grid, row_idx, col_idx = get_sampling_grid_test(grid_size, stride)
    visualize_sampling_grid(sampling_grid, row_idx, col_idx)
    print(f"Sampling grid shape: {sampling_grid.shape}, row_idx: {row_idx}, col_idx: {col_idx}")
    print("Sampling grid test completed.")
    plt.show()

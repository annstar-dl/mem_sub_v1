import numpy as np
from matplotlib import pyplot as plt
import os
import torch
from read_matlab import load_image_from_mat, load_image_from_mat_h5py
from membrane_subtract import membrane_subtract
from visualizations import visualize_3_images, visualize_2_images


def compare_reconstr_image(imgout, imgout_matlab,subtitle1, subtitle2, title="Reconstructed Image Comparison"):
    """
    Compare the reconstructed image from PyTorch with the one from MATLAB.

    Args:
        img (torch.Tensor): Original image tensor.
        imgout (torch.Tensor): Reconstructed image tensor from PyTorch.
        imgout_matlab (np.ndarray): Reconstructed image from MATLAB.
    """
    imgout_matlab = torch.tensor(imgout_matlab, device="cpu")
    imgout = imgout.to("cpu")  # Ensure imgout is on the CPU for comparison

    if torch.allclose(imgout, imgout_matlab, atol=1e-6):
        print("Reconstructed images are similar within the tolerance 1e-6.")

    diff = torch.abs(imgout - imgout_matlab)
    visualize_3_images(imgout, imgout_matlab, diff, subtitle1, subtitle2, "im1-im2", title)

def rms(tensor):
    """
    Calculate the root mean square of a tensor.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Root mean square of the input tensor.
    """
    return torch.sqrt(torch.mean(tensor ** 2))

def precentage_difference(imgout,imgout_matlab):
    rms_diff = rms(imgout - imgout_matlab)
    rms_matlab = rms(imgout_matlab)
    diff_precentage = rms_diff / rms_matlab
    print(f"RMS difference in reconstruction: {diff_precentage.item()}")

def visualize_image(img, title="Image", subtitle=""):
    """
    Visualize a single image tensor.

    Args:
        img (torch.Tensor): Image tensor to visualize.
        title (str): Title of the plot.
        subtitle (str): Subtitle of the plot.
    """

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.title(f"{title}\n{subtitle}")
    plt.axis('off')
    plt.colorbar()
def matlab_testing():
    maindir = r"/home/astar/Projects/matlab_code/data_from_matlab_code"
    file_path = os.path.join(maindir,r'fred_1.mat')
    img, mask, imgout_ht, imgout_sub_ht = load_image_from_mat(file_path, ["img","mask","mem", "sub"])
    mask = mask.astype(np.float64)
    img = img.astype(np.float64)
    imgout_my, imgout_sub_my = membrane_subtract(img,mask)


    compare_reconstr_image(imgout_sub_my, imgout_sub_ht,subtitle1="My",subtitle2="HT",
                           title="Reconstructed Image from me and matlab")
    precentage_difference(imgout_sub_my, torch.tensor(imgout_sub_ht))
    visualize_image(img, title="Original Image", subtitle="Image from MATLAB")




if __name__ == "__main__":
    # Example usage
    matlab_testing()
    plt.show()  # Show all plots at once
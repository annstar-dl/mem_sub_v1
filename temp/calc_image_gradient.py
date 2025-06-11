import matplotlib.pyplot as plt
from PIL import Image,ImageFilter
import os
from scipy.ndimage import convolve
import numpy as np

def laplacian_of_gaussian(image_path,output_path, sigma=1.0):
    """
    Apply Laplacian of Gaussian (LoG) to an image.

    Args:
        image_path (str): Path to the input image.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        Image: Processed image after applying LoG.
    """
    # Open the image
    image = Image.open(image_path).convert("L")  # Convert to grayscale

    # Create a Gaussian kernel
    kernel_size = int(6 * sigma + 1) | 1  # Ensure kernel size is odd
    gaussian_kernel = ImageFilter.GaussianBlur(radius=sigma)

    # Apply Gaussian blur
    blurred_image = image.filter(gaussian_kernel)

    # Apply Laplacian filter
    laplacian_kernel = ImageFilter.Kernel((3, 3), [-1, -1, -1, -1, 8, -1, -1, -1, -1], scale=1)
    log_image = blurred_image.filter(laplacian_kernel)
    log_image.save(output_path)


def image_gradient_direction(image_path, output_path):
    """
    Compute the gradient direction (in radians) for each pixel in a grayscale image.

    Args:
        image_path (str): Path to the input image.
        output_path (str, optional): Path to save the direction image (as grayscale visualization).

    Returns:
        np.ndarray: 2D array of gradient directions in radians.
    """
    image = Image.open(image_path).convert("L")
    img_np = np.array(image, dtype=np.float32)

    # Sobel kernels
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    # Convolve to get gradients

    Gx = convolve(img_np, Kx)
    Gy = convolve(img_np, Ky)

    # Compute gradient direction
    direction = np.arctan2(Gy, Gx)  # Radians, range [-pi, pi]

    if output_path:
        # Map direction to [0, 255] for visualization
        direction_vis = ((direction + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        Image.fromarray(direction_vis).save(output_path)

    return direction

def image_gradient_magnitude(image_path, output_path=None,smoothing_radius=2.0):
    """
    Compute the gradient magnitude for each pixel in a grayscale image.

    Args:
        image_path (str): Path to the input image.
        output_path (str, optional): Path to save the magnitude image (as grayscale visualization).

    Returns:
        np.ndarray: 2D array of gradient magnitudes.
    """
    image = Image.open(image_path).convert("L")
    # Apply Gaussian blur for denoising
    image = image.filter(ImageFilter.GaussianBlur(radius=(smoothing_radius,smoothing_radius)))
    img_np = np.array(image, dtype=np.float32)

    # Sobel kernels
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    # Convolve to get gradients
    Gx = convolve(img_np, Kx)
    Gy = convolve(img_np, Ky)

    # Compute gradient magnitude
    magnitude = np.sqrt(Gx**2 + Gy**2)

    if output_path:
        # Map magnitude to [0, 255] for visualization
        magnitude_vis = (magnitude / np.max(magnitude) * 255).astype(np.uint8)
        Image.fromarray(magnitude_vis).save(output_path)

    return magnitude

if __name__ == "__main__":
    # Example usage
    sigma = 1.0
    data_dir = r"/home/astar/Projects/vesicles_data"
    image_name = r"slot6_100_0002ms"
    image_path = os.path.join(data_dir,"test",image_name+".jpg")
    output_path = os.path.join(data_dir,"test_gradient_magnitude")  # Replace with your desired output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path, image_name + ".png")
    for smoothing_radius in [0.5, 1.0, 2.0, 3.0]:
        print(f"Calculating gradient magnitude with smoothing radius: {smoothing_radius}")
        magnitude = image_gradient_magnitude(image_path, None, smoothing_radius)
        plt.figure()
        plt.imshow(magnitude, cmap='gray')
        plt.title("Gradient magnitude, smoothing radius: {}".format(smoothing_radius))
    plt.show()
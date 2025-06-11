import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage.restoration import denoise_bilateral
from skimage import data, img_as_float

def read_image(image_path):
    """
    Read an image from the specified path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Image as a numpy array.
    """
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    return np.array(image, dtype=np.float32)

def calc_histogram(data, bins=256):
    """
    Calculate the histogram of an image.

    Args:
        image (np.ndarray): Input image as a numpy array.
        bins (int): Number of histogram bins.

    Returns:
        np.ndarray: Histogram of the image.
    """
    hist, _ = np.histogram(data, bins=bins, range=(0, 256))
    return hist

def visualize_histogram(hist, bins=256, title="Histogram"):
    """
    Visualize the histogram using matplotlib.

    Args:
        hist (np.ndarray): Histogram data.
        bins (int): Number of histogram bins.
    """
    hist = hist / hist.sum()  # Normalize the histogram
    plt.figure(figsize=(10, 5))
    plt.bar(range(bins), hist, width=1, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

def visualize_overlap_histograms(hist1, hist2, bins=256, title1="Histogram 1", title2="Histogram 2"):
    """
    Visualize two histograms on the same plot.

    Args:
        hist1 (np.ndarray): First histogram data.
        hist2 (np.ndarray): Second histogram data.
        bins (int): Number of histogram bins.
        title1 (str): Title for the first histogram.
        title2 (str): Title for the second histogram.
    """
    hist1 = hist1 / hist1.sum()  # Normalize the first histogram
    hist2 = hist2 / hist2.sum()  # Normalize the second histogram
    plt.figure(figsize=(10, 5))
    plt.bar(range(bins), hist1, width=1, color='blue', alpha=1.0, label=title1)
    plt.bar(range(bins), hist2, width=1, color='red', alpha=0.5, label=title2)
    plt.title("Overlap of " + title1 + " and " + title2)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()


def visualize_im(image, title=None):
    """
    Visualize an image using matplotlib.

    Args:
        image (np.ndarray): Image data as a numpy array.
        title (str, optional): Title for the plot.
    """
    plt.figure()
    plt.imshow(image, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')

def smooth_image(image, sigma=1.0):
    """
    Smooth an image using a Gaussian filter.

    Args:
        image (np.ndarray): Input image as a numpy array.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        np.ndarray: Smoothed image.
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(image, sigma=sigma)

def smooth_image_bilateral(image, sigma_color=1.0, sigma_spatial=5):
    """
    Smooth an image using a bilateral filter.

    Args:
        image (np.ndarray): Input image as a numpy array.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        np.ndarray: Smoothed image.
    """
    image = img_as_float(image.astype("uint8"))  # Convert image to float for processing
    denoised_image = denoise_bilateral(image, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
    denoised_image = (denoised_image * 255).astype(np.uint8)  # Convert back to uint8
    return denoised_image

def smooth_image_median(image, size=3):
    """
    Smooth an image using a median filter.

    Args:
        image (np.ndarray): Input image as a numpy array.
        size (int): Size of the median filter.

    Returns:
        np.ndarray: Smoothed image.
    """
    from scipy.ndimage import median_filter
    return median_filter(image, size=size)

def image_histogram(image_path,mask_path,output_path, bins=256):
    """
    Calculate the histogram of an image and visualize it.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the histogram image.
        bins (int): Number of histogram bins.
    """
    # Read the image
    image_org = read_image(image_path)
    image_org = image_org[160:-160, 160:-160] # Remove the border
    #image = smooth_image(image, sigma=2.0)  # Optional smoothing
    #image_org = smooth_image_bilateral(image_org, sigma_color=0.2,sigma_spatial=5)  # Optional smoothing with bilateral filter
    image_org = smooth_image_median(image_org, size=5)  # Optional smoothing with median filter
    visualize_im(image_org, title="Original Image")
    # Read membrane mask
    mask = read_image(mask_path)
    mask = mask[160:-160, 160:-160]  # Remove the border
    mask = mask > 0  # Convert mask to binary (0 or 1)
    nb_ones = np.sum(mask)
    # Visualize membrane histogram
    masked_data = image_org * mask  # Apply mask to the image data
    visualize_im(masked_data, title="Masked Image Data")
    masked_data = masked_data[mask>0] # Flatten the masked data
    print(f"Masked data contains {len(masked_data)} non-zero pixels., and {nb_ones} pixels in the mask")
    output_file = os.path.join(output_path, "membrane_"+image_name + ".png")
    hist_membrane = calc_histogram(masked_data, bins=bins)
    visualize_histogram(hist_membrane, bins=bins, title = "Membrane Histogram")
    plt.savefig(output_file)
    print(f"Histogram saved to {output_path}")

    # Read the non-membrane histogram
    image = image_org*(1-mask)  # Apply the inverse mask to get non-membrane data
    visualize_im(image, title="Image without membrane")
    image = image[(1-mask)>0]  # Flatten the non-membrane data
    # Calculate histogram
    hist_nonmembrane = calc_histogram(image, bins=bins)
    # Visualize the histogram
    visualize_histogram(hist_nonmembrane, bins=bins, title="Image without membrane")
    output_file = os.path.join(output_path, "nomembrane_" + image_name + ".png")
    plt.savefig(output_file)

    # Visualize the overlap of histograms
    visualize_overlap_histograms(hist_nonmembrane, hist_membrane, bins=bins, title1="Nonmembrane Histogram", title2="Membrane Histogram")

    class_im = classify_pixel_based_on_histogram(image_org, hist_membrane, hist_nonmembrane)
    visualize_im(class_im, title="Classified Image")
    plt.show()
def classify_pixel_based_on_histogram(image, hist1, hist2):
    """
    Classify pixels based on histogram values.

    Args:
        hist (np.ndarray): Histogram data.
        threshold (float): Threshold for classification.

    Returns:
        np.ndarray: Binary classification of pixels.
    """
    # Normalize the histogram
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    # Calculate the difference between histograms
    im_flat = image.flatten()
    im_flat = im_flat.astype(np.uint8)  # Ensure pixel values are in the range [0, 255]
    im_out = np.zeros_like(im_flat, dtype=np.uint8)
    hist1_val = hist1[im_flat]  # Get the histogram value for the pixel intensity
    hist2_val = hist2[im_flat]  # Get the histogram value for the pixel intensity
    im_out[np.where(hist1_val>hist2_val)] = 1  # Classify as non-membrane
    im_out = im_out.reshape(image.shape)  # Reshape back to original image shape
    # Classify pixels based on the threshold
    return im_out
if __name__ == "__main__":
    # Example usage
    data_dir = r"/home/astar/Projects/vesicles_data"
    image_name = r"slot6_100_0002ms"
    image_path = os.path.join(data_dir,"test",image_name+".jpg")
    mask_path = os.path.join(data_dir,"labels",image_name+".png")
    output_path = os.path.join(data_dir,"histograms")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    image_histogram(image_path, mask_path, output_path, bins=256)



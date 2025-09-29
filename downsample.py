import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from PIL import Image
from skimage import transform



def dowsample(img: np.ndarray, factor: int = 1) -> np.ndarray:
    """
    Downsample the data by a given factor.

    Args:
        data (np.ndarray): Input image.
        factor (int): Downsampling factor.

    Returns:
        np.ndarray: Downsampled data array.
    """
    height, width = img.shape[:2]
    new_width = int(width / factor)
    new_height = int(height / factor)
    if new_width < 1 or new_height < 1:
        raise ValueError("Downsampling factor is too large for the given data dimensions.")

    downsampled_data = fft_for_im(img, downsample=factor)
    return downsampled_data

def get_fft_center_coord(img: np.ndarray) -> np.ndarray:
    """
    Compute index of the 2D FFT the zero frequency component after fftshift.

    Args:
        img (np.ndarray): Input image array.

    Returns:
        (tuple):center_row, center_col
    """
    nb_rows = img.shape[0]
    nb_cols = img.shape[1]
    if nb_rows % 2 == 0:
        center_row = nb_rows // 2
    else:
        center_row = (nb_rows - 1) // 2

    if nb_cols % 2 == 0:
        center_col = nb_cols // 2
    else:
        center_col = (nb_cols - 1) // 2
    return center_row, center_col

def crop_im(img: np.ndarray, center_coord,crop_size) -> np.ndarray:
    """
    Crop im around the center coordinates.

    Args:
        img (np.ndarray): Input image array.
        center_coord (tuple): Center coordinates (center_row, center_col).
        crop_size (tuple): Size of the cropped image (downsample_row, downsample_col)

    Returns:
        np.ndarray: Cropped image.
    """
    img_crop = img[center_coord[0] - crop_size[0]//2:center_coord[0] + crop_size[0]//2,
             center_coord[1] - crop_size[1]//2:center_coord[1] + crop_size[1]//2]
    return img_crop

def fft_for_im(img: np.ndarray, downsample) -> np.ndarray:
    """
    Compute the 2D FFT of an image and shift the zero frequency component to the center.

    Args:
        img (np.ndarray): Input image array.

    Returns:
        np.ndarray: Shifted 2D FFT of the input image.
    """
    nb_rows, nb_cols = img.shape
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    nb_ds_rows, nb_ds_cols = int(nb_rows/downsample), int(nb_cols/downsample)
    center_row, center_col = get_fft_center_coord(img)
    fshift = crop_im(fshift, (center_row, center_col), (nb_ds_rows, nb_ds_cols))
    ifshift = np.fft.ifftshift(fshift)
    img = np.fft.ifft2(ifshift)
    img = img.real
    normalizing_value = (nb_ds_rows*nb_ds_cols)/(nb_rows*nb_cols)
    img = img*normalizing_value
    return img

def window_fun(img) -> np.ndarray:
    """
    Apply a low-pass Butterworth filter to the image.

    Args:
        img (np.ndarray): Input image array.
        cutoff (float): Cutoff frequency of the filter.
        fs (float): Sampling frequency of the image.
        order (int): Order of the Butterworth filter.

    Returns:
        np.ndarray: Filtered image.
    """
    # Generate a 2D Hann window
    window_x = signal.windows.hann(img.shape[1])
    window_y = signal.windows.hann(img.shape[0])
    window_2d = np.outer(window_y, window_x)
    return img
def draw_sign_wave(height, width, wavelength=100, amplitude=1, frequency=0.1, axis=0, value=1):
    """
    Draw a sine wave on the image.

    Args:
        img (np.ndarray): Input image array.
        amplitude (float): Amplitude of the sine wave.
        frequency (float): Frequency of the sine wave.
        axis (int): Axis along which to draw the sine wave (0 for rows, 1 for columns).
        value (float): Intensity value to set for the sine wave.

    Returns:
        np.ndarray: Image with the sine wave drawn on it.
    """
    x = np.arange(0, width, 1)
    grating = amplitude*np.sin(
        2 * np.pi * x/ wavelength
    )
    grating = grating[np.newaxis, :]
    grating = np.repeat(grating, height, axis=0)
    return grating

def visualize_im(img: np.ndarray, title: str = "Image"):
    """
    Visualize an image using matplotlib.

    Args:
        img (np.ndarray): Input image array.
        title (str): Title of the plot.

    Returns:
        None
    """
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

def downsample_test():
    img = draw_sign_wave(511, 1001, wavelength=1001/1)
    downsampled_img = dowsample(img, factor=5.6)
    print("Original shape:", img.shape)
    print("Downsampled shape:", downsampled_img.shape)
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.figure()
    plt.imshow(downsampled_img, cmap='gray')
    plt.title('Downsampled Image')

    #downsampled_img = dowsample(img - upsample_im, factor=8)
    #plt.figure()
    #plt.imshow(img - upsample_im, cmap='gray')
    #plt.title('Difference Image')

    plt.show()


if __name__ == "__main__":
    downsample_test()
import numpy as np
from matplotlib import pyplot as plt
from fuzzymask import fuzzymask
from new_downsample import down_sample


def converts_value_to_even(value: int) -> int:
    """
    Convert a value to the nearest even integer.

    Args:
        value (int): Input value.

    Returns:
        int: Nearest even integer.
    """
    if value % 2 != 0:
        value -= 1
    return value

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
    center_row = nb_rows // 2
    center_col = nb_cols // 2
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
    # shift the center coordinates to the bottom right corner of the cropped image
    # due to zero frequency component being in the middle of the FFT image
    img_crop = img[center_coord[0] - crop_size[0]//2:center_coord[0] + crop_size[0]//2+1,
             center_coord[1] - crop_size[1]//2:center_coord[1] + crop_size[1]//2+1]
    return img_crop

def downsample_with_fft(img: np.ndarray, downsample) -> np.ndarray:
    """
    Downsample image by cropping 2D FFT of an image.

    Args:
        img (np.ndarray): Input image array.

    Returns:
        np.ndarray: Shifted 2D FFT of the input image.
    """
    nb_rows, nb_cols = img.shape
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    visualize_im(fshift.real, "Shifted 2D FFT before cropping")
    print("Nonzero frequency component (DC):", np.where(fshift.real!=0))
    print(f"Original shape: {img.shape}")
    nb_ds_rows, nb_ds_cols = int(nb_rows/downsample), int(nb_cols/downsample)
    nb_ds_rows = converts_value_to_even(nb_ds_rows)
    nb_ds_cols = converts_value_to_even(nb_ds_cols)
    print(f"Nb downsampled rows: {nb_ds_rows}, Nb downsampled cols: {nb_ds_cols}")
    center_row, center_col = get_fft_center_coord(img)
    print(f"Center row: {center_row}, Center col: {center_col}")
    fshift = crop_im(fshift, (center_row, center_col), (nb_ds_rows, nb_ds_cols))
    ifshift = np.fft.ifftshift(fshift)
    img = np.fft.ifft2(ifshift)
    print(f"Downsampled shape: {img.shape}")
    scaler = ((nb_ds_rows+1)*(nb_ds_cols+1))/(nb_rows*nb_cols)
    img = scaler * img
    img = img.real
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


def visualize_im(img: np.ndarray, title: str = "Image", vmin=None, vmax=None) -> None:
    """
    Visualize an image using matplotlib.

    Args:
        img (np.ndarray): Input image array.
        title (str): Title of the plot.

    Returns:
        None
    """
    plt.figure()
    plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis('off')
def overlay_histograms(img1: np.ndarray, img2: np.ndarray, label1: str = "Image 1", label2: str = "Image 2") -> None:
    """
    Overlay histograms of two images.

    Args:
        img1 (np.ndarray): First image array.
        img2 (np.ndarray): Second image array.
        label1 (str): Label for the first image histogram.
        label2 (str): Label for the second image histogram.

    Returns:
        None
    """
    hist1, edges = np.histogram(img1.ravel(), bins=256, range=(0, 20))
    hist1 = hist1 / np.sum(hist1)
    hist2, _ = np.histogram(img2.ravel(), bins=256, range=(0, 20))
    hist2 = hist2 / np.sum(hist2)

    plt.figure()
    plt.plot(edges[:-1], hist1, label=label1)
    plt.plot(edges[:-1], hist2, label=label2)
    plt.legend(loc='upper right')
    plt.title("Overlayed Histograms")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()


def downsample_test():
    #img = draw_sign_wave(513, 1001, wavelength=1000/20)
    nb_samples = 41
    img = np.ones((nb_samples,nb_samples))
    freq = np.fft.fftfreq(nb_samples,1/nb_samples)
    print("Frequencies:", freq)
    freq_shift = np.fft.fftshift(freq)
    print("Where freq 0", np.where(freq_shift==0))
    print("Frequencies:", freq_shift)
    downsampled_img = downsample_with_fft(img, 4)
    visualize_im(img, title="Original Image")
    visualize_im(downsampled_img, title="Downsampled Image")

    print("Original shape:", img.shape)
    print("Downsampled shape:", downsampled_img.shape)
    print(f"Org mean: {np.mean(img)} , min: {np.min(img)}, max: {np.max(img)}")
    print(f"Downsample mean: {np.mean(downsampled_img)},"
        f" min: {np.min(downsampled_img)}, max: {np.max(downsampled_img)}")
    plt.show()

def downsample_real_test():
    from mrc_utils import load_mrc
    fpath = (r"/home/astar/Projects/vesicles_data/patri_subtractions/patri_02132024/patri_02132024/"
             r"FoilHole_14439761_Data_14437927_14437929_20231114_214246_fractions_aligned_mic_DW.mrc")
    img, _, _ = load_mrc(fpath)
    img = img[:,:-1]
    downsampled_img = downsample_with_fft(img, 4)
    new_size = (img.shape[0]//4, img.shape[1]//4)
    freq_mask = fuzzymask(new_size, r=0.45*np.array(new_size), risetime=0.05*new_size[0])
    downsampled_img_hemant_mask = down_sample(img, new_size, freq_mask)
    downsampled_img_hemant_no_mask = down_sample(img, new_size)
    print(f"Org mean: {np.mean(img)} , min: {np.min(img)}, max: {np.max(img)}")
    print(f"Downsample mean: {np.mean(downsampled_img)} , min: {np.min(downsampled_img)}, max: {np.max(downsampled_img)}")
    print("Original shape:", img.shape)
    print("Downsampled shape:", downsampled_img.shape)
    visualize_im(img, title="Original Image", vmin=np.min(img), vmax=np.max(img))
    visualize_im(downsampled_img, title="Downsampled Image",vmin=np.min(img), vmax=np.max(img))
    visualize_im(downsampled_img_hemant_mask, title="Downsampled Image Hemant",vmin=np.min(img), vmax=np.max(img))
    visualize_im(downsampled_img_hemant_no_mask - downsampled_img, title="Difference between two downsampled images",vmin=-1, vmax=1)
    overlay_histograms(downsampled_img,downsampled_img_hemant_no_mask, label1="My downsample", label2="Hemant's downsample")
    plt.show()


if __name__ == "__main__":

    downsample_real_test()
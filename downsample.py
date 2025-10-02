import numpy as np
from matplotlib import pyplot as plt
from fuzzymask import fuzzymask


def get_fft_center(n):
    """Return the center index for FFT operations."""
    return n//2

def get_start_stop_indices(n,m):
    """Get start and stop indices for cropping FFT."""
    offset=get_fft_center(n)-get_fft_center(m)
    left=offset
    right=m+offset
    return left,right

def down_sample(img,new_size, fuzzy_mask=None):
    """Downsample an image using FFT and an optional fuzzy mask."""
    #Get indices
    if fuzzy_mask is None:
        fuzzy_mask = 1 #no fuzzy mask
    n,m=img.shape
    center=(get_fft_center(n),get_fft_center(m))
    row_start, row_end = get_start_stop_indices(n, new_size[0])
    col_start, col_end = get_start_stop_indices(m, new_size[1])
    #Do the ffts and downsample
    x_fft = np.fft.fftshift(np.fft.fft2(img))
    x_ds_fft = x_fft[row_start:row_end, col_start:col_end]*fuzzy_mask
    x_ds = np.fft.ifft2(np.fft.ifftshift(x_ds_fft))
    x_ds = x_ds * np.prod(x_ds.shape) / np.prod(x_fft.shape)
    return np.real(x_ds)




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


def compare_to_matlab():
    import scipy.io as sio
    from matplotlib import pyplot as plt
    matfpath = r"/home/astar/Projects/Freds_code/micrograph3.mat"
    # Load the .mat file into a dictionary
    mat_data = sio.loadmat(matfpath)
    n = mat_data['n']
    n = np.squeeze(n)
    msk = mat_data['msk']
    im = mat_data['im3']
    im_ds_matlab = mat_data['out']
    msk_python = fuzzymask(n, r=0.45 * np.array(n), risetime=0.05 * n[0])
    im_ds_python = down_sample(im, n, fuzzy_mask=msk_python)
    visualize_im(im_ds_matlab, title="Downsampled Image Matlab")
    visualize_im(im_ds_python, title="Downsampled Image Python")
    visualize_im(im_ds_matlab-im_ds_python, title="Difference Image Matlab - Python", vmin=-0.1, vmax=0.1)
    print("Mean difference",np.mean(im_ds_matlab-im_ds_python))
    plt.show()


if __name__ == "__main__":
    compare_to_matlab()
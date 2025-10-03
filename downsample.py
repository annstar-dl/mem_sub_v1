import numpy as np


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



if __name__ == "__main__":
    pass
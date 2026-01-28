import numpy as np


def get_fft_center(n):
    """Return the center index for FFT operations."""
    return n//2

def get_start_stop_indices(n,m):
    """Get start and stop indices for cropping FFT.
    Args:
        n (int): Original size.
        m (int): New size.
    Returns:
        left (int): Start index.
        right (int): Stop index.
    """
    offset=get_fft_center(n)-get_fft_center(m)
    left=offset
    right=m+offset
    return left,right

def down_sample(img,new_shape, fuzzy_mask=None):
    """Downsample an image using FFT and an optional fuzzy mask."""
    #Get indices
    if fuzzy_mask is None:
        fuzzy_mask = 1 #no fuzzy mask

    row_start, row_end = get_start_stop_indices(img.shape[0], new_shape[0])
    col_start, col_end = get_start_stop_indices(img.shape[1], new_shape[1])
    #Do the ffts and downsample
    x_fft = np.fft.fftshift(np.fft.fft2(img))
    x_ds_fft = x_fft[row_start:row_end, col_start:col_end]*fuzzy_mask
    x_ds = np.fft.ifft2(np.fft.ifftshift(x_ds_fft))
    x_ds = x_ds * np.prod(x_ds.shape) / np.prod(x_fft.shape)
    return np.real(x_ds)

def up_sample(img, new_shape, fuzzy_mask=None):
    """Downsample an image using FFT and an optional fuzzy mask."""
    #Get indices
    if fuzzy_mask is None:
        fuzzy_mask = 1 #no fuzzy mask
    row_start, row_end = get_start_stop_indices(new_shape[0], img.shape[0])
    col_start, col_end = get_start_stop_indices(new_shape[1], img.shape[1])
    #Do the ffts and downsample
    x_fft = np.fft.fftshift(np.fft.fft2(img))
    x_fft = x_fft * fuzzy_mask
    # Create an array of zeros for the upsampled FFT
    upsampled_fft = np.zeros(new_shape, dtype=complex)
    upsampled_fft[row_start:row_end, col_start:col_end] = x_fft
    x_us = np.fft.ifft2(np.fft.ifftshift(upsampled_fft))
    x_us = x_us * np.prod(x_us.shape) / np.prod(x_fft.shape)
    return np.real(x_us)

if __name__ == "__main__":
    pass
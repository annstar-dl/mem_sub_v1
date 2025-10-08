import mrcfile
import numpy as np
from downsample import down_sample
from fuzzymask import fuzzymask
from math import ceil

MRC_MODE_DICT = {
    0: np.int8,
    1: np.int16,
    2: np.float32,
    3: np.complex64,
    4: np.complex128,
    6: np.uint16,
    12: np.float16,
    101: None,
}

FILE_TYPES = ["mrc", "st"]

def save_im_mrc_same_size(img, fpath, header):
    """Save the image to a MRC file keeping the same header information.
    Only the image data and statistic are changed.
    Args:
        img (numpy.ndarray): Image array to save.
        fpath (str): Path to save the image file.
        header (mrcfile.header): Header information to keep.
    Returns:
        None
    """
    with mrcfile.new(fpath, overwrite=True) as mrc_new:
        #Do not chnage dmin, dmax, dmean, and rms
        field_names = mrc_new.header.dtype.names
        for field in field_names:
            if field!="dmin" and field!="dmax" and field!="dmean" and field!="rms":
                mrc_new.header[field] = header[field]
        mrc_new.set_data(img.astype(np.float32))
        mrc_new.update_header_stats()

def load_mrc(in_file: str = None, transpose: tuple = None):
    """
    Load an MRC-like file (.mrc or .st) and return the data as a numpy array.

    Args:
        in_file (str): path to the MRC-like file (.mrc or .st - default: self.mrc_file)
        transpose (tuple): transpose the data array (default: no transpose)
        dawnsample_factor (int): factor by which to downsample the data (default: 1 - no downsampling)

    Returns:
        np.ndarray: data array of the MRC-like file
    """
    with mrcfile.open(in_file, permissive=True) as f:
        header = f.header
        data = f.data.astype(MRC_MODE_DICT[f.header["mode"].item()])
        voxel_size = f.voxel_size.item()
    #assert len(set(np.array(voxel_size).round(4))) == 1, "Voxel size must be isotropic"
    del in_file, f
    if data.ndim > 2:
        if data.shape[0] == 1:
            data = data[0]
        else:
            raise ValueError("Data has more than 2 dimensions, which is not supported.")
    if transpose is not None:
        data = data.transpose(transpose)
    return data, header, voxel_size

def calc_new_shape_based_on_voxel_size(old_shape,voxel_size,downsample_factor=None):
    """
    Calculate the downsampling factor based on the voxel size.

    Args:
        voxel_size (float): Original voxel size.
    Returns:
        int: Downsampling factor.
    """
    target_voxel_size = 4.5 # in Angstroms.
    # This is a voxel size that is approximately correspond to voxel
    # size of membrane detection training data
    if voxel_size <= 0 or target_voxel_size <= 0:
        raise ValueError("Voxel sizes must be positive values.")
    ds_factor = int(target_voxel_size / voxel_size)
    if downsample_factor is not None:
        ds_factor = downsample_factor
    if ds_factor < 1:
        ds_factor = 1
    old_shape = [next_multiple(old_shape[0],ds_factor), next_multiple(old_shape[1], ds_factor)]
    new_shape = int(old_shape[0]/ds_factor),int(old_shape[1]/ds_factor)
    return old_shape,new_shape, ds_factor

def downsample_mrc(data: np.ndarray, voxel_size: tuple) -> np.ndarray:
    """
    Downsample the MRC data based on the voxel size. If the voxel size is isotropic and greater than the target voxel size, downsample the data.
    Otherwise, return the data as is.

    Args:
        data (np.ndarray): Input image data.
        voxel_size (tuple): Voxel size in each dimension.

    Returns:
        np.ndarray: Downsampled image data.
        new_shape (tuple): New shape of the padded original image data.
    """
    if len(set(np.array(voxel_size[:2]).round(2))) != 1:
        raise ValueError("Voxel size must be isotropic for downsampling, but got: "
                         f"{voxel_size[0]:.2f}, {voxel_size[1]:.2f}")
    padded_shape, new_shape, ds_factor = calc_new_shape_based_on_voxel_size(data.shape, voxel_size[0], )
    print("Padded shape: {}, old shape: {}".format(padded_shape, data.shape))
    if ds_factor > 1:
        if np.any(np.array(padded_shape) > data.shape):
            print(f"Padding the image with shape {data.shape} to the new shape {padded_shape} before downsampling.")
            data = pad_im(data, padded_shape, padding_value=np.mean(data), mode="right_down")
        print(f"Downsampling factor  {ds_factor:.2f} is higher than 1, downsampling the data."
              f"Org voxel size: {voxel_size[0]:.2f} Å. Org data shape {data.shape} new data shape: {new_shape}")
        msk = fuzzymask(new_shape, r=0.48 * np.array(new_shape))
        data = down_sample(data, new_shape, fuzzy_mask=msk)
    return data

def pad_im(im, new_shape, padding_value, mode="right_down"):
    """Pad the image to the new shape with the given padding value.
    Args:
        im (numpy.ndarray): Image array to pad,
        new_shape (tuple): New shape of the image (height, width),
        padding_value (float): Value to use for padding,
        mode (str): Padding mode, either "right_down" or "center". Default is "right_down".
    Returns:
        numpy.ndarray: Padded image array.

        """

    pad_height = new_shape[0] - im.shape[0]
    pad_width = new_shape[1] - im.shape[1]

    if mode == "right_down":
        pad_top = 0
        pad_bottom = pad_height
        pad_left = 0
        pad_right = pad_width
    elif mode == "center":
        pad_top = int(pad_height // 2)
        pad_bottom = pad_height - pad_top
        pad_left = int(pad_width // 2)
        pad_right = pad_width - pad_left
    else:
        raise ValueError("Invalid padding mode. Supported modes are 'right_down' and 'center'.")
    padded_im = np.pad(im, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=padding_value)
    return padded_im

def convert_to_even(n):
    """Convert an integer to the nearest even integer."""
    return n if n % 2 == 0 else n + 1

def next_multiple(n, base):
    """Convert an integer to the next multiple of base."""

    n = n if n % base == 0 else ceil(n / base) * base
    n = convert_to_even(n/base)*base
    return int(n)
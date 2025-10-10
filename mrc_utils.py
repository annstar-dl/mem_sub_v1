import mrcfile
import numpy as np
from downsample import down_sample, up_sample
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
        #Do not change dmin, dmax, dmean, and rms
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

def calc_new_shape_based_on_voxel_size(old_shape,voxel_size,ds_factor=None):
    """
    Calculate the downsampling factor based on the voxel size.

    Args:
        voxel_size (float): Original voxel size.
        ds_factor (int, optional): Downsampling factor. If None, it will be calculated based on the voxel size.
    Returns:
        int: Downsampling factor.
    """
###!!!TODO: chose the target voxel size number in a range from 3.8 to 4.5 such that the padded size image multiple by ds_factor
    target_voxel_size = 4.5 # in Angstroms.
    # This is a voxel size that is approximately correspond to voxel
    # size of membrane detection training data
    if voxel_size <= 0 or target_voxel_size <= 0:
        raise ValueError("Voxel sizes must be positive values.")

    if ds_factor is not None:
        ds_factor = ds_factor
    else:
        ds_factor = int(target_voxel_size / voxel_size)
    if ds_factor < 1:
        ds_factor = 1
    old_shape = [next_multiple(old_shape[0],ds_factor), next_multiple(old_shape[1], ds_factor)]
    new_shape = int(old_shape[0]/ds_factor),int(old_shape[1]/ds_factor)
    return old_shape,new_shape, ds_factor

def downsample_mrc(data: np.ndarray, voxel_size: tuple,downsample_factor=None) -> np.ndarray:
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
    padded_org_shape, ds_shape, ds_factor = calc_new_shape_based_on_voxel_size(data.shape, voxel_size[0],downsample_factor)
    print("Padded shape: {}, old shape: {}".format(padded_org_shape, data.shape))
    if ds_factor > 1:
        if np.any(np.array(padded_org_shape) > data.shape):
            print(f"Padding the image with shape {data.shape} to the new shape {padded_org_shape} before downsampling.")
            data = pad_im(data, padded_org_shape, padding_value=0, mode="center")
        print(f"Downsampling factor  {ds_factor:.2f} is higher than 1, downsampling the data."
              f"Org voxel size: {voxel_size[0]:.2f} Å. Org data shape {data.shape} new data shape: {ds_shape}")
        msk = fuzzymask(ds_shape, r=0.48 * np.array(ds_shape))
        data = down_sample(data, ds_shape, fuzzy_mask=msk)
    return data

def upsample_mrc_to_original(img_ds, original_shape, voxel_size, downsample_factor=None):
    """
    Upsample the downsampled image to the original shape using nearest neighbor interpolation.
    Args:
        img_ds (numpy.ndarray): Downsampled image array.
        original_shape (tuple): Original shape of the image (height, width).
        padded_org_shape (tuple): Padded original shape of the image (height, width).
    Returns:
        numpy.ndarray: Upsampled image array.
    """
    padded_org_shape,new_shape, ds_factor = calc_new_shape_based_on_voxel_size(original_shape, voxel_size[0], downsample_factor)
    fuzzy_mask = fuzzymask(img_ds.shape, r=0.48 * np.array(img_ds.shape))
    #upsampled_img = up_sample(img_ds, original_shape, fuzzy_mask)
    upsampled_img = up_sample(img_ds, padded_org_shape, fuzzy_mask)
    if np.any(np.array(padded_org_shape) > np.array(original_shape)):
        print(f"Cropping the upsampled image with shape {upsampled_img.shape} to the original shape {original_shape}.")
        upsampled_img = crop_im(upsampled_img, original_shape, mode="center")
    return upsampled_img

def pad_im(im, new_shape, padding_value, mode="right_down"):
    """
    Pad the image to the new shape with the given padding value.
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

def crop_im(im, new_shape, mode="right_down"):
    """
    Crop the image to the new shape.
    Args:
        im (numpy.ndarray): Image array to crop,
        new_shape (tuple): New shape of the image (height, width),
        mode (str): Cropping mode, either "right_down" or "center". Default is "right_down".
    Returns:
        numpy.ndarray: Cropped image array.
    """
    if im.shape[0] < new_shape[0] or im.shape[1] < new_shape[1]:
        raise ValueError("New shape must be smaller than the original shape.")
    if mode == "right_down":
        cropped_im = im[:new_shape[0], :new_shape[1]]
    elif mode == "center":
        start_y = (im.shape[0] - new_shape[0]) // 2
        start_x = (im.shape[1] - new_shape[1]) // 2
        cropped_im = im[start_y:start_y + new_shape[0], start_x:start_x + new_shape[1]]
    else:
        raise ValueError("Invalid cropping mode. Supported modes are 'right_down' and 'center'.")
    return cropped_im

def next_multiple(n, base):
    """Convert an integer to the next multiple of base."""
    n = n if n % base == 0 else ceil(n / base) * base
    return int(n)

if __name__ == "__main__":
    pass
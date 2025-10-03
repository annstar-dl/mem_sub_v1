import mrcfile
import numpy as np
from temp.new_downsample import down_sample
from fuzzymask import fuzzymask

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
def save_im_mrc(img, fpath, header):
    """Save the image to a MRC file.
    Args:
        img (numpy.ndarray): Image array to save.
        fpath (str): Path to save the image file.
    """
    with mrcfile.new(fpath, overwrite=True) as mrc_new:
        #Do not change nx,ny,nz values in the header
        #Do not chnage dmin, dmax, dmean, and rms
        field_names = mrc_new.header.dtype.names
        for field in field_names:
            if field!="nx" and field!="ny" and field!="nz" and field!="dmin" and field!="dmax" and field!="dmean" and field!="rms":
                mrc_new.header[field] = header[field]
        mrc_new.set_data(img.astype(np.float32))

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

def calc_downsampling_factor_based_on_voxel_size(voxel_size):
    """
    Calculate the downsampling factor based on the voxel size.

    Args:
        voxel_size (float): Original voxel size.
    Returns:
        int: Downsampling factor.
    """
    target_voxel_size = 4.2 # in Angstroms.
    # This is a voxel size that is approximately correspond to voxel
    # size of membrane detection training data
    if voxel_size <= 0 or target_voxel_size <= 0:
        raise ValueError("Voxel sizes must be positive values.")
    downsample_factor = target_voxel_size / voxel_size
    if downsample_factor < 1:
        downsample_factor = 1
    return downsample_factor

def downsample_mrc(data: np.ndarray, voxel_size: tuple) -> np.ndarray:
    """
    Downsample the MRC data based on the voxel size. If the voxel size is isotropic and greater than the target voxel size, downsample the data.
    Otherwise, return the data as is.

    Args:
        data (np.ndarray): Input image data.
        voxel_size (tuple): Voxel size in each dimension.

    Returns:
        np.ndarray: Downsampled image data.
    """
    if len(set(np.array(voxel_size[:2]).round(2))) != 1:
        raise ValueError("Voxel size must be isotropic for downsampling, but got: "
                         f"{voxel_size[0]:.2f}, {voxel_size[1]:.2f}")
    downsample_factor = calc_downsampling_factor_based_on_voxel_size(voxel_size[0])
    if downsample_factor > 1:
        new_shape = (int(data.shape[0]/downsample_factor), int(data.shape[1]/downsample_factor))
        msk = fuzzymask(new_shape, r=0.45 * np.array(new_shape), risetime=0.05 * new_shape[0])
        data = down_sample(data, new_shape, fuzzy_mask=msk)
    return data
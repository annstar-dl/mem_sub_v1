import numpy as np
import mrcfile as mrc
from skimage import transform

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

def load_mrc(in_file: str = None, transpose: tuple = None, downsample_factor: int = 4) -> np.ndarray:
    """
    Load an MRC-like file (.mrc or .st) and return the data as a numpy array.

    Args:
        in_file (str): path to the MRC-like file (.mrc or .st - default: self.mrc_file)
        transpose (tuple): transpose the data array (default: no transpose)

    Returns:
        np.ndarray: data array of the MRC-like file
    """
    with mrc.open(in_file, permissive=True) as f:
        data = f.data.astype(MRC_MODE_DICT[f.header["mode"].item()])
        voxel_size = f.voxel_size.item()
    #assert len(set(np.array(voxel_size).round(4))) == 1, "Voxel size must be isotropic"
    del in_file, f, voxel_size
    if data.ndim > 2:
        if data.shape[0] == 1:
            data = data[0]
        else:
            raise ValueError("Data has more than 2 dimensions, which is not supported.")
    if transpose is not None:
        data = data.transpose(transpose)
    if downsample_factor > 1:
        data = dowsample(data, downsample_factor)  # Downsample the data by a factor of 4
    return data

def dowsample(data: np.ndarray, factor: 4) -> np.ndarray:
    """
    Downsample the data by a given factor.

    Args:
        data (np.ndarray): Input data array.
        factor (int): Downsampling factor.

    Returns:
        np.ndarray: Downsampled data array.
    """
    height, width = data.shape[:2]
    new_width = int(width / factor)
    new_height = int(height / factor)
    downsampled_img = transform.resize(image=data, output_shape=(new_height, new_width), order=1, mode='reflect', anti_aliasing=True)
    return downsampled_img

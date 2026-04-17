import mrcfile
import numpy as np
from mem_sub.mrc_tools.downsample import down_sample, up_sample
from mem_sub.mrc_tools.fuzzymask import fuzzy_disk, fuzzy_rectangle

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
            raise ValueError(f"Data has {data.shape[2]} channels, which is not supported.")
    if transpose is not None:
        data = data.transpose(transpose)
    return data, header, voxel_size

def new_shape_mrc_downsampling(old_shape,voxel_size,ds_factor=None):
    """
    Calculate the downsampling factor based on the voxel size.

    Args:
        voxel_size (float): Original voxel size.
        ds_factor (int, optional): Downsampling factor. If None, it will be calculated based on the voxel size.
    Returns:
        padded_shape (tuple): New shape of the padded original image data.
        new_shape (tuple): New shape of the downsampled image data.
        ds_factor (int): Downsampling factor.
    """
    target_voxel_size = 4.0 # in Angstroms.
    # This is a voxel size that is approximately correspond to voxel
    # size of membrane detection training data
    if voxel_size <= 0 or target_voxel_size <= 0:
        raise ValueError("Voxel sizes must be positive values.")

    if ds_factor is not None:
        ds_factor = ds_factor
    else:
        #get the integer downsampling factor(rounding down)
        # TODO: change int to round to get the closest downsampling factor to the target voxel size
        ds_factor = round(target_voxel_size / voxel_size)
    if ds_factor < 1:
        ds_factor = 1
    cropped_shape = [croped_value_even_multiple_of_ds_factor(old_shape[0],ds_factor), croped_value_even_multiple_of_ds_factor(old_shape[1], ds_factor)]

    return cropped_shape,ds_factor

def downsample_micrograph(data: np.ndarray, voxel_size: float, border, cropping_mode="center", return_logs=False,
                          subtract_mean=True) -> np.ndarray:
    """
    Downsample the MRC data based on the voxel size. To prevent iliasing artifacts, the image is multiplied
    by a fuzzy rectangle mask before downsampling that brings the signal to zero at the borders.
    If the voxel size is isotropic and greater than the target voxel size, downsample the data.
    Otherwise, return the data as is.

    Args:
        data (np.ndarray): Input image data.
        voxel_size (tuple): Voxel size in each dimension.
        border (int): Border size for the fuzzy mask. Default is 0.
        cropping_mode (str): Padding mode, either "right_down" or "center". Default is "center".
        return_logs (bool, optional): Whether to return the logs.
        subtract_mean (bool, optional): Whether to subtract the mean value from the data.
        Do not subtract mean if downsampling labels!

    Returns:
        np.ndarray: Downsampled image data.
    """
    # Find new shape for downsampling based on voxel size and calculate downsampling factor
    # Assume voxel_size is isotropic
    # Recalculate the new shape for downsampling considering that after the downsampling the
    # image shape should be multiple of downsampling factor
    downsampling_log = {"org_voxel_size":voxel_size,"ds_voxel_size":voxel_size,"ds_factor":1,"org_shape":data.shape,
                        "ds_shape":data.shape,"cropping_mode":cropping_mode, "cropped_shape":data.shape,
                        "mean_of_uncropped_image":float(data.mean()), "ds_fuzzy_border_size":border}
    cropped_shape, ds_factor = new_shape_mrc_downsampling(data.shape, voxel_size)

    if ds_factor > 1:
        downsampling_log["ds_factor"] = ds_factor
        downsampling_log["cropped_shape"] = cropped_shape

        if np.any(np.array(cropped_shape) < data.shape):
            print(f"Cropping the image with shape {data.shape} to the new shape {cropped_shape} before downsampling.")
            data = crop_im(data, cropped_shape, mode=cropping_mode)

        data_mean = np.mean(data)
        downsampling_log["mean_of_cropped_image"] = float(data_mean)
        # subtract mean before downsampling
        if subtract_mean:
            print("Mean is subtracted during downsampling")
            data = data - data_mean
        # Apply fuzzy rectangle mask to the data to reduce edge artifacts
        if border > 0:
            # making signal to look periodic to reduce edge artifacts during downsampling
            # by applying fuzzy rectangle mask that smoothly goes to zero at the borders
            # the border size is scaled according to the downsampling factor
            # the border size is usually set to the radius of the neighborhood used in membrane estimation
            # algorithm
            fuzzy_rec = fuzzy_rectangle(shape=data.shape, border=border * ds_factor)
            data = data * fuzzy_rec
        ds_shape = (data.shape[0] // ds_factor, data.shape[1] // ds_factor)
        downsampling_log["ds_shape"] = ds_shape
        downsampling_log["ds_voxel_size"] = ds_factor * voxel_size
        print(f"Downsampling factor  {ds_factor:.2f} is higher than 1, downsampling the data."
              f"Org data shape {data.shape} new data shape: {ds_shape}")
        print(f"Org voxel size: {voxel_size:.3f} Å. Downsampled voxel size: {voxel_size*ds_factor} Å.")

        # Create fuzzy disk mask for downsampling to reduce aliasing artifacts
        # why 0.48? because the fuzzy disk radius should be less than half of the image size
        # to avoid edge artifacts during downsampling
        msk = fuzzy_disk(ds_shape, r=0.48 * np.array(ds_shape))
        data = down_sample(data, ds_shape, fuzzy_mask=msk)
        if subtract_mean:
            data = data + data_mean
    if return_logs:
        return data, downsampling_log
    else:
        return data

def upsample_micrograph(img_ds, original_shape, voxel_size, padding_mode="center") -> np.ndarray:
    """
    Upsample the downsampled image to the original shape using nearest neighbor interpolation.
    Args:
        img_ds (numpy.ndarray): Downsampled image array.
        original_shape (tuple): Original shape of the image (height, width).
        voxel_size (tuple): Voxel size in each dimension.
        padding_mode (str): Padding mode, either "right_down" or "center". Default is "center".
    Returns:
        numpy.ndarray: Upsampled image array.
    """
    cropped_shape, ds_factor = new_shape_mrc_downsampling(original_shape, voxel_size)
    msk = fuzzy_disk(img_ds.shape, r=0.48 * np.array(img_ds.shape))
    upsampled_img = up_sample(img_ds, cropped_shape, msk)
    if np.any(np.array(cropped_shape) < np.array(original_shape)):
        print(f"Padding the upsampled image with shape {upsampled_img.shape} to the original shape {original_shape}.")
        upsampled_img = pad_im(upsampled_img, original_shape, padding_value=0, mode=padding_mode)
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

def croped_value_even_multiple_of_ds_factor(value, ds_factor):
    """
    Crop the value to the nearest even multiple of the downsampling factor.
    The value should be even after divading by the downsampling factor, due to the FFT algorithm
     used during downsampling.
    Args:
        value (int): Value to crop,
        ds_factor (int): Downsampling factor.
    Returns:
        int: Cropped value.
    """
    if value <=0:
        raise ValueError("Value must be greater than 0.")
    if ds_factor <= 0:
        raise ValueError("Downsampling factor must be greater than 0.")
    return (value // (2*ds_factor)) *(2* ds_factor)

if __name__ == "__main__":
    pass
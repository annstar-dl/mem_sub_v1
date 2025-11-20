import numpy as np
from downsample import down_sample
from fuzzymask import fuzzy_disk, fuzzy_rectangle
from mrc_utils import new_shape_mrc_downsampling, pad_im
import argparse
import os
from run_mrc_subtraction import read_img, save_im
from sub_utils import read_dict_from_yaml_file
from matplotlib import pyplot as plt

def downsample_micrograph(data: np.ndarray,voxel_size: float, border = 0,padding_mode = "center") -> np.ndarray:
    """
    Downsample the MRC data based on the voxel size. If the voxel size is isotropic and greater than the target voxel size, downsample the data.
    Otherwise, return the data as is.

    Args:
        data (np.ndarray): Input image data.
        pad_org_shape (tuple): New shape of the padded original image data.
        voxel_size (tuple): Voxel size in each dimension.
        border (int): Border size for the fuzzy mask. Default is 0.
        padding_mode (str): Padding mode, either "right_down" or "center". Default is "center".

    Returns:
        np.ndarray: Downsampled image data.
    """

    pad_org_shape, ds_shape, ds_factor = new_shape_mrc_downsampling(data.shape, voxel_size)
    if ds_factor > 1:
        if border > 0:
            fuzzy_rec = fuzzy_rectangle(shape=data.shape, border=border * ds_factor)
            data = data * fuzzy_rec
        if np.any(np.array(pad_org_shape) > data.shape):
            print(f"Padding the image with shape {data.shape} to the new shape {pad_org_shape} before downsampling.")
            data = pad_im(data, pad_org_shape, padding_value=0, mode=padding_mode)
        print(f"Downsampling factor  {ds_factor:.2f} is higher than 1, downsampling the data."
              f"Org data shape {data.shape} new data shape: {ds_shape}")

        msk = fuzzy_disk(ds_shape, r=0.48 * np.array(ds_shape))
        data = down_sample(data, ds_shape, fuzzy_mask=msk)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--output_path", type=str, default="/home/astar/Projects/vesicles_data/vb_flagella_png_sub/images_jpg/vb_flagella_png_ds",
                        help="Directory path containing folders with images and labels")
    parser.add_argument("-ip", "--imgs_path", default="/home/astar/Projects/vesicles_data/vb_flagella_png_sub/vb_flagella_png", type=str, help="Directory path containing mrc micrographs")
    parser.add_argument("-vs", "--voxel", default=1.068, type=float, )
    args = parser.parse_args()
    parameters = read_dict_from_yaml_file()
    border = parameters["r"]
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    for filename in os.listdir(args.imgs_path):
        img = read_img(os.path.join(args.imgs_path, filename))
        for i in range(4):
            plt.figure()
            plt.imshow(img[:,:,i], cmap='gray')
            plt.title(f'Original Image Channel {i}')
        plt.show()

        img = downsample_micrograph(img, args.voxel,border, "center")
        print("Shape of ds data:", img.shape)
        save_im(img, os.path.join(args.output_path,filename))
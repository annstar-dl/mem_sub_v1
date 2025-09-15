#!/usr/bin/env python3
#
#   mrc2tif.py - Convert MRC-like files to TIFF or JPEG file formats
#   author: Christopher JF Cameron


import argparse
import mrcfile as mrc
import numpy as np
import os
from PIL import Image
from glob import glob
from tqdm import tqdm
from skimage import io, transform

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

def load_mrc(in_file: str = None, transpose: tuple = None, downsample_factor: int = 1):
    """
    Load an MRC-like file (.mrc or .st) and return the data as a numpy array.

    Args:
        in_file (str): path to the MRC-like file (.mrc or .st - default: self.mrc_file)
        transpose (tuple): transpose the data array (default: no transpose)

    Returns:
        np.ndarray: data array of the MRC-like file
    """
    with mrc.open(in_file, permissive=True) as f:
        header = f.header
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

    return data, header

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

def main(args: argparse.Namespace) -> None:
    """
    Convert MRC files to TIFF and JPEG file formats.

    Args:
        args (argparse.Namespace): command-line arguments

    Returns:
        None
    """
    #   get list of MRC-like files
    files = []
    for ext in FILE_TYPES:
        files.extend(glob(os.path.join(args.in_dir, f"*.{ext}")))

    #   convert MRC-like files to TIFF and JPEG file formats
    for file in tqdm(files):
        #   read in the MRC-like file data
        data,_ = load_mrc(file, downsample_factor=args.downsample_factor)

        #   save as TIFF image
        basename, _ = os.path.splitext(os.path.basename(file))
        if args.format == "tif":
            data_tif = data
            Image.fromarray(data_tif).save(os.path.join(args.out_dir, f"{basename}.tif"))
        elif args.format == "jpeg" or args.format == "jpg":
            if args.scale:
                data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
            #   save as JPEG image
            # Normalize the data to the range [0, 255] for JPEG saving
            # tiff.imwrite(
            #     os.path.join(args.out_dir, f"{basename}.tif"),
            #     cv2.normalize(data, None, 0, 1, cv2.NORM_MINMAX),
            #     photometric="minisblack",
            # )

            data_jpg = data.astype(np.uint8)
            io.imsave(os.path.join(args.out_dir, f"{basename}.{args.format}"), data_jpg)
        elif args.format == "png":
            data_png = data.astype(np.uint8)
            print("Data range", np.min(data_png), np.max(data_png))
            io.imsave(os.path.join(args.out_dir, f"{basename}.{args.format}"), data_png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MRC-like files to TIFF and JPEG file formats")
    parser.add_argument(
        "in_dir",
        type=str,
        help="Directory path to input directory containing MRC-like files",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=None,
        help="Directory path to output directory ",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["tif", "jpeg", "jpg", "png"],
        default="jpeg",
        help="Output file format (default: jpeg)",
    )
    parser.add_argument(
        "-ds",
        "--downsample_factor",
        type=int,
        default=4,
        help="Factor by which to downsample the images (default: 4)",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Scale images to 0-255 for JPEG output (default: False)",
    )
    args = parser.parse_args()

    assert os.path.isdir(args.in_dir), f"Input directory does not exist: {args.in_dir}"
    data_dir_name = os.path.basename(os.path.normpath(args.in_dir))
    args.out_dir = os.path.join(args.out_dir, data_dir_name +"_"+args.format+"_"+f"ds{args.downsample_factor}")
    if args.out_dir is None:
        # set output directory to input directory if not specified
        args.out_dir = args.in_dir
    else:
        # create output directory if it does not exist
        os.makedirs(args.out_dir, exist_ok=True)

    if args.format not in ["tif", "jpeg", "jpg", "png"]:
        parser.error(f"Unsupported format: {args.format}. Supported formats are 'tif' and 'jpeg'.")
    if args.downsample_factor > 1:
        print(f"Downsample images by {args.downsample_factor}.")
    main(args)

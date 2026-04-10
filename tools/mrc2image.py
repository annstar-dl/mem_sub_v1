import argparse
import numpy as np
import os
import json
from PIL import Image
from glob import glob
from tqdm import tqdm
from skimage import io
from mem_sub.mrc_tools.mrc_utils import load_mrc, FILE_TYPES, downsample_micrograph
from mem_sub.membrane_est.utils import read_parameters_from_yaml_file


def save_json(logs: dict, log_path: str) -> None:
    with open(log_path, "w") as f:
        json.dump(logs,f)

def convert_dir(args: argparse.Namespace) -> None:
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
        files.extend(glob(os.path.join(args.mrc_path, f"*.{ext}")))

    #   convert MRC-like files to TIFF and JPEG file formats
    for file_path in tqdm(files):
        print(f"Processing {file_path}")
        #   read in the MRC-like file data
        args.file_path = file_path
        convert_file(args)

def convert_file(args: argparse.Namespace) -> None:
    data, _, voxel_size = load_mrc(args.file_path)
    #  downsample the data if the voxel size is greater than the target voxel size

    if args.downsampling_allowed:
        if args.border_size==-1:
            parameters = read_parameters_from_yaml_file()
            border = parameters["r"]
            print(f"Setting border size to {border}")
        else:
            border = args.border_size
        data, logs = downsample_micrograph(data, voxel_size[0], border, "center", return_logs=True,
                                           subtract_mean=args.sub_mean)
        # save downsampling parameters to json
        logs_path = os.path.join(args.logs_dir, os.path.splitext(os.path.basename(args.file_path))[0] + ".json")
        save_json(logs, logs_path)
    basename, _ = os.path.splitext(os.path.basename(args.file_path))
    # stretch contrast
    if args.scale:
        # clip values to the minimum inside the border
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
        data = data.astype(np.uint8)
    # save as tif
    if args.format == "tif":
        Image.fromarray(data).save(os.path.join(args.out_dir, f"{basename}.tif"))
    # save as jpg or png
    elif args.format == "jpeg" or args.format == "jpg" or args.format == "png":
        io.imsave(os.path.join(args.out_dir, f"{basename}.{args.format}"), data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MRC-like files to TIFF and JPEG file formats")
    parser.add_argument(
        "mrc_path",
        type=str,
        help="Directory or file path to input containing MRC micrograph",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=None,
        help="Directory path to output directory ",
    )
    parser.add_argument(
        "-f","--format",
        type=str,
        choices=["tif", "jpeg", "jpg", "png"],
        default="jpeg",
        help="Output file format (default: jpeg)",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Scale images to 0-255 for JPEG output (default: False)",
    )
    parser.add_argument("-dsa",
        "--downsampling_allowed",
        action="store_true",
        help="Allow downsampling based on voxel size (default: False)",
    )
    parser.add_argument(
        "--sub_mean",
        action="store_true",
        help="Allow to subtract mean from image during downsampling (default: False). Do not subtract mean if downsampling labels!",
    )
    parser.add_argument(
        "-bs",
        "--border_size", type=int,
        help="Downsampling fuzzy mask size. A smoothing mask is applied to an image to make signal go to zero"
            "at the border. The border_size is a size of fuzzy border in downsampled image."
            "If this value set to -1 the border would be set to parameter r from parameters.yml file.")
    args = parser.parse_args()


    print(f"Output will be saved to: {args.out_dir}")
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    args.logs_dir = os.path.join(args.out_dir, "logs")
    if args.downsampling_allowed:
        print("Downsampling based on voxel size is allowed.")
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    if args.format not in ["tif", "jpeg", "jpg", "png"]:
        parser.error(f"Unsupported format: {args.format}. Supported formats are 'tif' and 'jpeg'.")

    if os.path.isdir(args.mrc_path):
        convert_dir(args)
    else:
        if not args.mrc_path.lower().endswith(tuple(FILE_TYPES)):
            raise ValueError(f"File {args.mrc_path} is not an MRC file.")
        args.file_path = args.mrc_path
        convert_file(args)

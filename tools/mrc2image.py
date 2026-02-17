import argparse
import numpy as np
import os
from PIL import Image
from glob import glob
from tqdm import tqdm
from skimage import io
from mem_sub.mrc_tools.mrc_utils import load_mrc, FILE_TYPES, downsample_micrograph
from mem_sub.membrane_est.utils import read_parameters_from_yaml_file


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
        files.extend(glob(os.path.join(args.in_dir, f"*.{ext}")))

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
        if args.use_border:
            parameters = read_parameters_from_yaml_file()
            border = parameters["r"]
        else:
            border = 0
        data = downsample_micrograph(data, voxel_size[0], border, "center")
    # save as TIFF image
    basename, _ = os.path.splitext(os.path.basename(args.file_path))

    if args.scale:
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255

    if args.format == "tif":
        Image.fromarray(data).save(os.path.join(args.out_dir, f"{basename}.tif"))
    elif args.format == "jpeg" or args.format == "jpg":
        #   save as JPEG image
        # Normalize the data to the range [0, 255] for JPEG saving
        # tiff.imwrite(
        #     os.path.join(args.out_dir, f"{basename}.tif"),
        #     cv2.normalize(data, None, 0, 1, cv2.NORM_MINMAX),
        #     photometric="minisblack",
        # )
        data = data.astype(np.uint8)
        io.imsave(os.path.join(args.out_dir, f"{basename}.{args.format}"), data)
    elif args.format == "png":
        data = data.astype(np.uint8)
        io.imsave(os.path.join(args.out_dir, f"{basename}.{args.format}"), data)

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
        "-fn",
        "--file_name",
        type=str, default=None,
    help="Name of file to convert (default: None), if None process all files in the folder",
    )
    parser.add_argument("-bd","--use_border", help="User border during downsampling as we do in membrane estimation", action="store_true")
    args = parser.parse_args()

    assert os.path.isdir(args.in_dir), f"Input directory does not exist: {args.in_dir}"
    data_dir_name = os.path.basename(os.path.normpath(args.in_dir))
    args.out_dir = os.path.join(args.out_dir,data_dir_name +"_"+ args.format+ "_ds" if args.downsampling_allowed else data_dir_name)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.downsampling_allowed:
        print("Downsampling based on voxel size is allowed.")
    if args.format not in ["tif", "jpeg", "jpg", "png"]:
        parser.error(f"Unsupported format: {args.format}. Supported formats are 'tif' and 'jpeg'.")
    if args.file_name is None:
        convert_dir(args)
    else:
        if not args.file_name.lower().endswith(tuple(FILE_TYPES)):
            raise ValueError(f"File {args.file_name} is not an MRC file.")

        args.file_path = os.path.join(args.in_dir, args.file_name)
        convert_file(args)

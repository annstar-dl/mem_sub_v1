import argparse
import os
from os.path import basename

import numpy as np
from skimage import io

def find_green_pixels(image):
    """
    Find pixels in the image that are predominantly green.

    Args:
        image (numpy.ndarray): Input image array.

    Returns:
        numpy.ndarray: Binary mask of green pixels.
    """

    # Calculate the green channel
    green_channel = image[:, :, 1]
    green_pixels = green_channel > 100  # Threshold for green pixels
    return green_pixels

def main(args):
    """
    Convert color images to binary format by thresholding.

    Args:
        args (argparse.Namespace): command-line arguments
    """
    input_dir = args.input_dir
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir),"labels")
    else:
        output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(input_dir, filename)
            img = io.imread(img_path)
            membrane_mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
            membrane_mask[find_green_pixels(img)] = 1
            basename, _ = os.path.splitext(filename)
            output_path = os.path.join(output_dir, basename +".png")
            io.imsave(output_path, membrane_mask)
            print(f"Converted {filename} to binary format and saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert color images to binary format.")
    parser.add_argument("input_dir", type=str, help="Directory containing input images.")
    parser.add_argument("-o","--output_dir", type=str,default=None, help="Directory containing output images.")
    args = parser.parse_args()

    main(args)
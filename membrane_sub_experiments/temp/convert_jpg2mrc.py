#!/usr/bin/env python3
"""
convert_jpg2mrc.py

Reads a JPG (or other image formats supported by Pillow) and writes it as a single-slice
MRC file. The script stores voxel size information in the MRC header. By default the
voxel-size is a 3-tuple with the first element 4.5 as requested.

Usage example:
    python temp/convert_jpg2mrc.py input.jpg output.mrc --voxel-size 4.5 4.5 4.5

Dependencies:
    pip install mrcfile pillow numpy

Behavior details / assumptions:
- The image is converted to grayscale (single channel) before storing in the MRC.
- The produced MRC has shape (nz=1, ny=height, nx=width).
- voxel_size is interpreted as (vx, vy, vz) in the same order as (x, y, z).
  The header.cella (physical map size) is set to (nx*vx, ny*vy, nz*vz).

"""

import argparse
import os
import sys
from typing import Sequence

try:
    from PIL import Image
except Exception as e:
    print("Missing dependency: Pillow is required (pip install pillow).", file=sys.stderr)
    raise

try:
    import numpy as np
except Exception as e:
    print("Missing dependency: numpy is required (pip install numpy).", file=sys.stderr)
    raise

try:
    import mrcfile
except Exception as e:
    print("Missing dependency: mrcfile is required (pip install mrcfile).", file=sys.stderr)
    raise


def parse_voxel_size(tokens: Sequence[str]):
    vals = [float(t) for t in tokens]
    if len(vals) != 3:
        raise ValueError("voxel-size must contain exactly 3 float values (vx vy vz)")
    return tuple(vals)


def convert_image_to_mrc(infile: str, outfile: str, voxel_size=(4.5, 1.0, 1.0)):
    """Read image, convert to grayscale, write MRC with header voxel size.

    Args:
        infile: path to image file (jpg, png, ...)
        outfile: path where .mrc will be written (will overwrite)
        voxel_size: tuple (vx, vy, vz) in the same units you want stored in the header.
                    The first element is set to 4.5 by default.
    """
    if not os.path.exists(infile):
        raise FileNotFoundError(f"Input file not found: {infile}")

    im = Image.open(infile)
    # convert to single-channel (grayscale) to produce a 3D single-slice map
    im = im.convert("L")
    arr = np.asarray(im)

    # Ensure data is C-contiguous and a supported dtype
    data = np.asarray(arr, dtype=np.float32, order="C")

    # MRC expects data shape (nz, ny, nx)
    data3 = data[np.newaxis, ...]  # nz=1

    nz, ny, nx = data3.shape

    # Write MRC and set header cell dimensions based on voxel size
    with mrcfile.new(outfile, overwrite=True) as mrc:
        # mrc.set_data wants a numpy array with the correct dtype
        mrc.set_data(data3.astype(np.float32))

        # cella = (nx*vx, ny*vy, nz*vz)
        vx, vy, vz = voxel_size
        # header.cella expects a sequence of 3 floats
        mrc.header.cella = (float(nx) * float(vx), float(ny) * float(vy), float(nz) * float(vz))

        # For convenience also store voxel size as an attribute on the MRC object (not standard
        # in the MRC header but useful when re-reading via mrcfile)
        try:
            mrc.voxel_size = voxel_size
        except Exception:
            # Older mrcfile versions may not have a voxel_size attribute; ignore silently.
            pass

        mrc.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert a JPG (or other image) to a single-slice MRC with voxel size in header.")
    parser.add_argument("infile", help="Input image file (jpg/png/...)")
    parser.add_argument("outfile", help="Output .mrc file")
    parser.add_argument("--voxel-size", nargs=3, metavar=("VX", "VY", "VZ"),
                        help="Voxel size as three floats: VX VY VZ. Default: 4.5 4.5 4.5",
                        default=("4.5", "4.5", "4.5"))

    args = parser.parse_args(argv)

    vs = parse_voxel_size(args.voxel_size)
    try:
        convert_image_to_mrc(args.infile, args.outfile, voxel_size=vs)
        print(f"Wrote MRC to {args.outfile} with voxel_size={vs}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()


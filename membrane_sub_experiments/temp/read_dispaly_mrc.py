#!/usr/bin/env python3
"""
read_dispaly_mrc.py

Read an MRC file and visualize its image data.
- Prints basic header info (shape, voxel size if present, cella)
- If the map is 3D, you can select a slice with --slice (default: middle)
- Saves visualization to a PNG when --out is provided; otherwise attempts to show interactively

Usage examples:
    python temp/read_dispaly_mrc.py input.mrc --slice 5 --out preview.png
    python temp/read_dispaly_mrc.py input.mrc

This script uses mrcfile and matplotlib. Install if missing:
    pip install mrcfile matplotlib numpy

"""

from __future__ import annotations

import argparse
import sys
import os
from typing import Optional, Tuple, Any

try:
    import mrcfile
except Exception as e:
    print("Missing dependency: mrcfile (pip install mrcfile)", file=sys.stderr)
    raise

try:
    import numpy as np
except Exception as e:
    print("Missing dependency: numpy (pip install numpy)", file=sys.stderr)
    raise

# Matplotlib: use Agg backend by default so the script works in headless environments when saving.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    print("Missing dependency: matplotlib (pip install matplotlib)", file=sys.stderr)
    raise


def read_mrc(path: str) -> Tuple[np.ndarray, Any]:
    """Read an MRC file and return the data array and opened mrc object.

    Returns a tuple (data, mrc) where `data` is a numpy array (dtype depending on file)
    and `mrc` is the opened mrcfile object (still open, caller may read header attributes).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    m = mrcfile.open(path, permissive=True)
    # m.data returns a memory-mapped or in-memory numpy array view
    data = m.data.copy() if hasattr(m.data, "copy") else np.array(m.data)
    return data, m


def print_header_info(m: Any) -> None:
    hdr = m.header
    try:
        shape = m.data.shape
    except Exception:
        shape = None

    print("MRC header info:")
    if shape is not None:
        print(f"  data shape: {shape}")
    try:
        # mrcfile exposes header.cella and header.map if present
        cella = tuple(hdr.cella)
        print(f"  header.cella (physical map size): {cella}")
    except Exception:
        pass

    # voxel size may be stored as attribute if written by mrcfile convenience code
    try:
        vs = getattr(m, "voxel_size")
        print(f"  voxel_size attribute: {vs}")
    except Exception:
        pass

    # Print a small selection of header fields if available
    try:
        print(f"  map: {hdr.map}")
    except Exception:
        pass


def choose_slice_index(data: np.ndarray, requested: Optional[int]) -> int:
    """Return a valid slice index for 3D data. If data is 2D returns 0.

    If `requested` is None, return the middle slice (nz//2). Clips out-of-range requests.
    """
    if data.ndim == 2:
        return 0
    if data.ndim == 3:
        nz = data.shape[0]
        if requested is None:
            idx = nz // 2
        else:
            idx = int(requested)
        # clip
        if idx < 0:
            idx = 0
        if idx >= nz:
            idx = nz - 1
        return idx
    raise ValueError("Data must be 2D or 3D")


def visualize(data: np.ndarray, slice_idx: Optional[int], cmap: str = "gray", vmin: Optional[float] = None, vmax: Optional[float] = None, out: Optional[str] = None) -> None:
    """Visualize the MRC data. If `out` is provided saves to that path; otherwise tries plt.show().

    For 3D data, uses the selected slice (first axis is z).
    """
    if data.ndim == 3:
        idx = choose_slice_index(data, slice_idx)
        image = data[idx, :, :]
        print(f"Displaying slice {idx} of {data.shape[0]}")
    elif data.ndim == 2:
        image = data
        print("Displaying 2D image")
    else:
        raise ValueError("Unsupported data ndim: expected 2 or 3")

    # Auto scale if vmin/vmax unspecified (percentile-based to avoid outlier clipping)
    if vmin is None or vmax is None:
        p1, p99 = np.percentile(image, (1, 99))
        if vmin is None:
            vmin = float(p1)
        if vmax is None:
            vmax = float(p99)

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("MRC image")

    if out:
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {out}")
        plt.close()
    else:
        # Try to show interactively; if backend doesn't support it (e.g. 'Agg'), fallback to saving a temporary PNG
        # Detect whether the current backend is interactive. Calling plt.show() with a
        # non-interactive backend (like 'Agg') triggers a UserWarning but not an Exception,
        # so we check the backend explicitly and save a temporary image instead.
        try:
            backend = matplotlib.get_backend()
            interactive_backend = matplotlib.is_interactive() and not backend.lower().startswith("agg")
        except Exception:
            backend = None
            interactive_backend = False

        if interactive_backend:
            plt.show()
        else:
            import tempfile

            fd, tmp = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            plt.savefig(tmp, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Interactive display not available (backend={backend}); saved temporary preview to {tmp}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read and visualize an MRC file (single slice or 2D)")
    p.add_argument("infile", help="Path to input .mrc file")
    p.add_argument("--slice", type=int, default=None, help="Slice index for 3D volumes (default: middle)")
    p.add_argument("--out", help="If set, save visualization to this PNG path instead of showing")
    p.add_argument("--cmap", default="gray", help="Matplotlib colormap (default: gray)")
    p.add_argument("--vmin", type=float, default=None, help="Lower bound for intensity scaling")
    p.add_argument("--vmax", type=float, default=None, help="Upper bound for intensity scaling")
    return p


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    data, m = read_mrc(args.infile)
    print_header_info(m)

    try:
        visualize(data, args.slice, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax, out=args.out)
    finally:
        try:
            m.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

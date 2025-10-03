from scipy.ndimage import binary_dilation
from skimage import io
import numpy as np
import os


def dilate_mask(mask: np.ndarray, dilation_radius: int) -> np.ndarray:
    """ Dilate a binary mask using a circular structuring element."""
    # Create a sample NumPy array from the PIL image
    img_array = np.array(mask) > 0

    # Perform the binary dilation using a simple structuring element
    for _ in range(dilation_radius):
        img_array = binary_dilation(img_array)

    return img_array

if __name__ == "__main__":
    maindir = r"/home/astar/Projects/vesicles_data/patri_subtractions/labels"
    dilatedir = r"/home/astar/Projects/vesicles_data/patri_subtractions/labels_dilated"
    os.makedirs(dilatedir, exist_ok=True)
    for fname in os.listdir(maindir):
        if fname.endswith(".png"):
            fpath = os.path.join(maindir, fname)
            mask = io.imread(fpath, as_gray=True)
            dilated_mask = dilate_mask(mask, dilation_radius=10)
            io.imsave(os.path.join(dilatedir,fname), dilated_mask.astype(np.uint8))
from kornia.geometry import scale

from mrc_utils import load_mrc, downsample_micrograph, upsample_micrograph
from fuzzymask import fuzzy_disk, fuzzy_rectangle
from membrane_estimation import add_border_to_mask
from bg_estimation import get_background
import numpy as np
from matplotlib import pyplot as plt
from scipy import io
from skimage import io as ioim
from sampling_grid import  get_sampling_grid, select_points_within_boundary

if __name__ == "__main__":
    fpath_org = r"/home/astar/Projects/vesicles_data/from_Fred/subtracted/20211122/20211122/slot6_100_0002.mrc"
    fpath_mask = r"/home/astar/Projects/vesicles_data/from_Fred/subtracted_new_fuzrec/20211122/labels/slot6_100_0001.png"
    data, _, voxel_size = load_mrc(fpath_org)
    mask = ioim.imread(fpath_mask)
    mask = (mask > 0).astype(np.float64)
    data = np.ones_like(data)
    print("Original shape: {}".format(data.shape))
    print("Original data range: min {}, max {}".format(np.min(data), np.max(data)))
    print("Data std: {}, mean: {}".format(np.std(data), np.mean(data)))
    border = 4*20
    rec = fuzzy_rectangle(shape=data.shape, border=border)
    data = data * rec
    plt.figure()
    plt.imshow(data, cmap='gray')
    plt.title('Data after applying fuzzy rectangle mask')
    plt.axis('off')
    data_bordered = add_border_to_mask(data, border)
    plt.figure()
    plt.imshow(data_bordered, cmap='gray')
    plt.title('difference after adding border to data')
    plt.axis('off')
    plt.show()



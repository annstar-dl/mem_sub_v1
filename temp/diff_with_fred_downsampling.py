from kornia.geometry import scale

from mrc_utils import load_mrc, downsample_mrc
from fuzzymask import fuzzymask
from downsample import down_sample
import numpy as np
from matplotlib import pyplot as plt
from scipy import io

if __name__ == "__main__":
    fpath_org = r"/home/astar/Projects/vesicles_data/from_Fred/subtracted/20211122/20211122/slot6_100_0002.mrc"

    data, _, voxel_size = load_mrc(fpath_org)
    print("Original shape: {}".format(data.shape))
    print("Original data range: min {}, max {}".format(np.min(data), np.max(data)))
    print("Data std: {}, mean: {}".format(np.std(data), np.mean(data)))
    scaled_data = (data - np.mean(data))
    print("Scaled data range: min {}, max {}".format(np.min(scaled_data), np.max(scaled_data)))
    my_ds = downsample_mrc(data, voxel_size)
    my_fuzzy_mask = fuzzymask(my_ds.shape, r=0.48 * np.array(my_ds.shape))
    #fpath_ds = r"/home/astar/Projects/vesicles_data/from_Fred/subtracted/20211122/20211122/slot6_100_0002ms.mrc"
    #fred_ds, _, _ = load_mrc(fpath_ds)
    freds_fuzzy_mask_path = r"/home/astar/Projects/Freds_code/fz_mask.mat"
    freds_fuzzy_mask = io.loadmat(freds_fuzzy_mask_path)['msk']
    fred_ds_path = r"/home/astar/Projects/Freds_code/ds_slot6_100_0002.mat"
    fred_ds = io.loadmat(fred_ds_path)['mOut']
    #compare fuzzy masks
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(freds_fuzzy_mask - my_fuzzy_mask, cmap='gray')
    plt.title('Difference between Fred\'s fuzzy mask and My fuzzy mask')
    plt.subplot(1, 3, 2)
    plt.imshow(freds_fuzzy_mask, cmap='gray')
    plt.title('Fred\'s fuzzy mask')
    plt.subplot(1, 3, 3)
    plt.imshow(my_fuzzy_mask, cmap='gray')
    plt.title('My fuzzy mask')
    #plt.colorbar()

    print("Fred's data range: min {}, max {}".format(np.min(fred_ds), np.max(fred_ds)))
    print("My downsampled data range: min {}, max {}".format(np.min(my_ds), np.max(my_ds)))
    plt.figure(figsize=(12, 6))
    #fred_ds_normal = (fred_ds - np.min(fred_ds)) / (np.max(fred_ds) - np.min(fred_ds))
    #my_ds_normal = (my_ds - np.min(my_ds)) / (np.max(my_ds) - np.min(my_ds))
    plt.imshow(fred_ds - my_ds, cmap='gray')
    plt.title('Difference between Fred\'s downsampled and My downsampled')
    plt.colorbar()


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(fred_ds, cmap='gray')
    plt.title('Fred\'s Downsampled Image')
    plt.subplot(1, 2, 2)
    plt.imshow(my_ds, cmap='gray')
    plt.title('My Downsampled Image')

    plt.figure()
    plt.plot(my_ds[500, :], label='My Downsampled', alpha=0.7)
    plt.plot(fred_ds[500, :], label='Fred\'s Downsampled', alpha=0.7)

    plt.title('Row 500 Comparison')
    plt.xlabel('Pixel Index')
    plt.ylabel('Normalized Intensity')
    plt.legend()
    plt.show()

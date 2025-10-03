from mrc_utils import load_mrc
from matplotlib import pyplot as plt
import numpy as np
from skimage import io
from new_downsample import visualize_im
def find_membrane_from_subtraction(org_img, sub_img):


    if org_img.shape != sub_img.shape:
        raise ValueError("Original and subtracted images must have the same shape.")
    membrane = org_img - sub_img
    return membrane

if __name__ == "__main__":
    org_path = r"/home/astar/Projects/vesicles_data/patri_subtractions/patri_02132024/FoilHole_14439761_Data_14437927_14437929_20231114_214246_fractions_aligned_mic_DW.mrc"
    sub_path_10_dilation = r"/home/astar/Projects/vesicles_data/patri_subtractions/reconstructions_10_dilation/subtracted_mrc/FoilHole_14439761_Data_14437927_14437929_20231114_214246_fractions_aligned_mic_DW.mrc"
    sub_path_10_dilation_longer = r"/home/astar/Projects/vesicles_data/patri_subtractions/reconstructions/subtracted_mrc/FoilHole_14439761_Data_14437927_14437929_20231114_214246_fractions_aligned_mic_DW.mrc"
    start_row, end_row = 3446, 3728
    start_col, end_col = 1638, 1860
    org_img, _, _ = load_mrc(org_path)
    sub_img_10_dilation, _, _ = load_mrc(sub_path_10_dilation)
    sub_img_10_dilation_longer, _, _ = load_mrc(sub_path_10_dilation_longer)
    #org_img = org_img[start_row:end_row, start_col:end_col]
    #sub_img = sub_img[start_row:end_row, start_col:end_col]
    membrane_10_dilation = find_membrane_from_subtraction(org_img, sub_img_10_dilation)
    membrane_10_dilation_longer = find_membrane_from_subtraction(org_img, sub_img_10_dilation_longer)
    visualize_im(org_img, title="Original Image", vmin=8, vmax=10)
    visualize_im(membrane_10_dilation, title="Membrane Image 10 dilation")
    visualize_im(membrane_10_dilation_longer, title="Membrane Image 10 dilation longer")
    visualize_im(sub_img_10_dilation, title="Subtracted Image 10 dilation", vmin=8, vmax=10)
    visualize_im(sub_img_10_dilation_longer, title="Subtracted Image 10 dilation longer", vmin=8, vmax=10)
    plt.show()

from membrane_sub.mrc_tools.mrc_utils import load_mrc
from matplotlib import pyplot as plt
import numpy as np
from new_downsample import visualize_im
def find_membrane_from_subtraction(org_img, sub_img):


    if org_img.shape != sub_img.shape:
        raise ValueError("Original and subtracted images must have the same shape.")
    membrane = org_img - sub_img
    return membrane

def patri_img():
    org_sub_path = r"/home/astar/Projects/vesicles_data/patri_test_crop/subtracted/test_crop.mrc"
    sub_sub_path = r"/home/astar/Projects/vesicles_data/patri_test_crop/subtracted/reconstructions/subtracted_mrc/test_crop.mrc"
    org_path = r"/home/astar/Projects/vesicles_data/patri_test_crop/org/test_crop.mrc"
    sub_path = r"/home/astar/Projects/vesicles_data/patri_test_crop/org/reconstructions/subtracted_mrc/test_crop.mrc"
    org_sub_img, _, _ = load_mrc(org_sub_path)
    sub_sub, _, _ = load_mrc(sub_sub_path)
    org_img, _, _ = load_mrc(org_path)
    sub, _, _ = load_mrc(sub_path)
    #org_img = org_img[start_row:end_row, start_col:end_col]
    #sub_img = sub_img[start_row:end_row, start_col:end_col]
    membrane_sub = find_membrane_from_subtraction(org_sub_img, sub_sub)
    membrane = find_membrane_from_subtraction(org_img, sub)
    corr = np.corrcoef(membrane_sub[650,:], membrane[650,:])
    print("Correlation between membranes: {}".format(corr))


    visualize_im(membrane_sub, title="Membrane sub")
    visualize_im(membrane, title="Membrane")
    #membrane = membrane/np.mean(np.abs(membrane))
    #membrane_sub = membrane_sub / np.mean(np.abs(membrane_sub))

    plt.figure(figsize=(12, 6))
    plt.plot(membrane[650, :], label='Membrane Original', alpha=0.7)
    plt.plot(membrane_sub[650, :], label='Membrane Subtracted', alpha=0.7)
    plt.title('Membrane Profiles Comparison')
    plt.xlabel('Pixel Index')
    plt.ylabel('Normalized Intensity')
    plt.legend()

    plt.figure()
    #membrane[500, :] = -1
    plt.imshow(membrane, cmap='gray')
    plt.title('Membrane Original')

    plt.figure()
    plt.imshow(membrane+membrane_sub, cmap='gray')
    plt.title('Membrane Original + Membrane Subtracted')

    plt.show()

def freds_img_test():
    org_path = r"/home/astar/Projects/vesicles_data/from_Fred/subtracted/20211122/20211122/slot6_100_0001.mrc"
    sub_path = r"/home/astar/Projects/vesicles_data/from_Fred/subtracted/20211122/reconstructions/subtracted_mrc/slot6_100_0001.mrc"
    org_img, old_header, old_voxel_size = load_mrc(org_path)
    sub, new_header, new_voxel_size = load_mrc(sub_path)
    membrane = find_membrane_from_subtraction(org_img, sub)
    visualize_im(membrane, title="Membrane")
    #compare headers:
    for key in old_header.dtype.names:
        if key in new_header.dtype.names:
            if np.any(old_header[key] != new_header[key]):
                print(f"Header key {key} differs: old {old_header[key]}, new {new_header[key]}")
        else:
            print(f"Header key {key} not in new header")
    print(f"Old voxel size: {old_voxel_size}, New voxel size: {new_voxel_size}")


    plt.show()

if __name__ == "__main__":
    freds_img_test()
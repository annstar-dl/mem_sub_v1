from get_basis_test import visualize_3_images
import torch
from matplotlib import pyplot as plt
import time
from sampling_grid import get_sampling_grid
from membrane_subtract import membrane_subtract
from basis_fn import get_basis, get_basis_sequential, get_w_function
from testing_with_matlab import compare_angles
from fit_basis_to_data import fit_basis_to_data
import scipy

def load_image_from_mat(file_path, variable_name='image'):
    """
    Load an image from a .mat file.

    Args:
        file_path (str): Path to the .mat file.
    Returns:
        np.ndarray: The loaded image as a NumPy array.
    """
    mat_data = scipy.io.loadmat(file_path)
    if variable_name not in mat_data:
        raise KeyError(f"Variable '{variable_name}' not found in the .mat file.")
    return mat_data[variable_name]
def set_mask_boundary_to_zero(mask):
    mask[:35]=0
    mask[-35:]=0
    mask[:,:35]=0
    mask[:,-35:]=0
    return mask
def subtrack_membrane_test():
    # Extract a small patch and its mask
    r = 20
    nb_iter = 1

    file_path = r'/home/astar/Projects/data_from_matlab_code/mk_1.mat'
    img = load_image_from_mat(file_path, "img")
    mask = load_image_from_mat(file_path, "mask")
    img = torch.tensor(img, dtype=torch.float64)
    mask= torch.tensor(mask, dtype=torch.float32)
    mask=set_mask_boundary_to_zero(mask)
    img = img - torch.mean(img)  # Center the patch around zero

    file_path = r"/home/astar/Projects/data_from_matlab_code/mk_1_thetas.mat"
    angles_matlab = load_image_from_mat(file_path, "theta_array")
    mask, row_idx, col_idx = get_sampling_grid(mask, 4, 4)  # Get the sampling grid from the mask
    file_path = r"/home/astar/Projects/data_from_matlab_code/mk1_x.mat"
    row_idx = load_image_from_mat(file_path, "x")-1
    row_idx = torch.tensor(row_idx).squeeze().to(torch.int64)
    file_path = r"/home/astar/Projects/data_from_matlab_code/mk1_y.mat"
    col_idx = load_image_from_mat(file_path, "y")-1
    col_idx = torch.tensor(col_idx).squeeze().to(torch.int64)
    dataimg = img.clone()
    basis = get_basis(dataimg, mask, row_idx, col_idx, r,)
    imgout = fit_basis_to_data(img, basis, row_idx, col_idx, r)
    #compare_angles(angles,angles_matlab)

    #imgout, mask = membrane_subtract(img,mask,r, nb_iter)
    visualize_3_images(img,imgout,img - imgout*mask, "Org","Membrane", "Subtr")
    #file_path = r"/home/astar/Projects/data_from_matlab_code/mk_1_2.mat"
    #imgout_matlab = load_image_from_mat(file_path,"mem")
    #imgout_matlab = imgout_matlab[top:top+rows,left:left+cols]
    #imgout_matlab = torch.tensor(imgout_matlab)
    #visualize_3_images(imgout,imgout_matlab, imgout-imgout_matlab, "Pytorch membrane",
    #                   "Matlab membrane", "Pytorch-Matlab")

if __name__=="__main__":
    subtrack_membrane_test()
    plt.show()

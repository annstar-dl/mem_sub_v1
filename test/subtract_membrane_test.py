from get_basis_test import visualize_3_images
import torch
from matplotlib import pyplot as plt
import time
from testing_with_matlab import load_image_from_mat
from membrane_subtract import membrane_subtract
from basis_fn import get_basis


def subtrack_membrane_test():
    # Extract a small patch and its mask
    r = 20
    nb_iter = 1
    top, left = 700, 300
    rows = 300
    cols = 300
    file_path = r'/home/astar/Projects/data_from_matlab_code/mk_1.mat'
    #file_path = r'/home/astar/Projects/matlab_code/mem_data_1.mat'  # Replace with your .mat file path
    img = load_image_from_mat(file_path, "img")
    mask = load_image_from_mat(file_path, "mask")
    img, mask = img[top:top+rows,left:left+cols],mask[top:top+rows,left:left+cols]
    img = torch.tensor(img, dtype=torch.float64)
    mask= torch.tensor(mask, dtype=torch.float32)
    img = img - torch.mean(img)  # Center the patch around zero
    #basis, angles = get_basis(img,mask,row_idx, col_idx,r,True)
    #file_path = r"/home/astar/Projects/data_from_matlab_code/mem_data_1_thetas.mat"
    #angles_matlab = load_image_from_mat(file_path, "mem")

    imgout, mask = membrane_subtract(img,mask,r, nb_iter)
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

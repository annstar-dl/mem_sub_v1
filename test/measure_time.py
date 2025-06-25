import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import time
from sampling_grid import get_sampling_grid, select_points_within_boundary
from read_matlab import load_image_from_mat

from basis_fn import get_basis
from fit_basis_to_data import fit_basis_to_data

def measure_time(run_gd=False):

    r = 20  # Radius of neighborhood
    maindir = r"/home/astar/Projects/data_from_matlab_code"
    file_path = os.path.join(maindir,r'mk_1.mat')
    img, mask = load_image_from_mat(file_path, ["img","mask"])
    img = torch.tensor(img, dtype=torch.float64)
    mask= torch.tensor(mask, dtype=torch.float64)
    img = img - torch.mean(img)  # Center the patch around zero
    mask, row_idx, col_idx = get_sampling_grid(mask, 4, 4)  # Get the sampling grid from the mask
    row_idx, col_idx = select_points_within_boundary(img, r, row_idx, col_idx)
    dataimg = img.detach().clone()
    basis = get_basis(dataimg, mask, row_idx, col_idx, r)
    if run_gd:
        imgout = fit_basis_to_data(img, basis, row_idx, col_idx, r, 0.025, 1)

if __name__ == "__main__":
    print("Is cuda available:", torch.cuda.is_available())
    start_time = time.time()
    measure_time()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
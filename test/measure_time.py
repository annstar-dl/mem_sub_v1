import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import time
from sampling_grid import get_sampling_grid, select_points_within_boundary
from read_matlab import load_image_from_mat
from torch.profiler import profile, record_function, ProfilerActivity
from basis_fn import get_basis
from fit_basis_to_data import fit_basis_to_data, fit_basis_to_data_batched


def measure_time(run_gd=False):

    r = 20  # Radius of neighborhood
    step = 4  # Step size for sampling grid
    maindir = r"/home/astar/Projects/data_from_matlab_code"
    file_path = os.path.join(maindir,r'mk_1.mat')
    img, mask = load_image_from_mat(file_path, ["img","mask"])
    img = torch.tensor(img, dtype=torch.float64)
    mask= torch.tensor(mask, dtype=torch.float64)
    img = img - torch.mean(img)  # Center the patch around zero
    mask, row_idx, col_idx = get_sampling_grid(mask, 4, step)  # Get the sampling grid from the mask
    row_idx, col_idx = select_points_within_boundary(img, r, row_idx, col_idx)
    dataimg = img.detach().clone()
    basis = get_basis(dataimg, mask, row_idx, col_idx, r)
    if run_gd:
        imgout = fit_basis_to_data_batched(img, basis, row_idx, col_idx, r, 0.025, 30, step)

def run_with_profiler():
    activities = [ProfilerActivity.CPU]
    sort_by_keyword = "cpu" + "_time_total"
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
        sort_by_keyword = "cuda" + "_time_total"
    with profile(activities=activities, record_shapes=False) as prof:
        with record_function("bases rotations"):
            measure_time(True)
    print(prof.key_averages(group_by_input_shape=True).table(sort_by=sort_by_keyword, row_limit=10))
    prof.export_chrome_trace("trace.json")
if __name__ == "__main__":
    print("Is cuda available:", torch.cuda.is_available())
    start_time = time.time()
    measure_time(True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

from mem_sub.membrane_est.align_image import rotate_images_kornia, calculate_mse_loss
from mem_sub.membrane_est.basis_fn import get_radius_of_inner_circle, get_basis,get_w_function
from mem_sub.membrane_est.sub_utils import get_patches_from_image_adv_indexing, creat_idx_batches_for_parl_sum, add_patches_to_image_batched
from matplotlib import pyplot as plt
from mem_sub.membrane_est.utils import read_parameters_from_yaml_file, read_img, save_im
from mem_sub.membrane_est.basis_fn import create_gaussian_disc
from  mem_sub.membrane_est.membrane_estimation import prepare_micrograph, fit_membrane
import torch
from mem_sub.mrc_tools.mrc_utils import load_mrc, downsample_micrograph
from mem_sub.metrics.dog import subtraction_metric
import numpy as np


def get_membrane_blending(dataimg, mask_blured, row_idx, col_idx, r, step):
    """Get the membrane profile at each grid point.
    Args:
        dataimg (torch.Tensor): The image from previous processing step of shape (H,W).
        row_idx (torch.Tensor): X coordinates of grid of shape (N), number of grid points.
        col_idx (torch.Tensor): Y coordinates of grid of shape (N), number of grid points.
        r (int): Radius of a neighbourhood.
        step (int): Step size for creating batches of non-overlapping patches."""
    r_in = get_radius_of_inner_circle(r)  # Get the radius of the inner circle
    basis, theta = get_basis(img_tensor, row_idx, col_idx,parameters["r"])
    _, gauss = create_gaussian_disc(2 * [(2 * r_in + 1)], r_in)

    gauss = gauss.to(basis.device)  # Move the Gaussian weights to the same device as the basis
    #basis = basis * gauss.unsqueeze(0)  # Apply Gaussian weights to the basis functions
    batched_row_idxs, batched_col_idxs, bases_idxs = creat_idx_batches_for_parl_sum(row_idx, col_idx, r_in, step)
    ones = torch.ones_like(dataimg, dtype=torch.float64)  # Create a tensor of ones with the same shape as the basis
    domains = get_patches_from_image_adv_indexing(ones, r_in, row_idx, col_idx)
    empty_img = torch.zeros_like(dataimg)
    if torch.cuda.is_available():
        domains = domains.to("cuda")
        empty_img = empty_img.to("cuda")
    domains = domains * gauss.unsqueeze(0)


    imgout = add_patches_to_image_batched(basis, empty_img.detach().clone(), r_in, batched_row_idxs, batched_col_idxs,
                                          bases_idxs)
    domains_sum = add_patches_to_image_batched(domains,empty_img.detach().clone(), r_in, batched_row_idxs, batched_col_idxs,
                                          bases_idxs)
    imgout/=(domains_sum+1e-6)  # Normalize the output image by the sum of the domains to get the average membrane profile
    mask_blured = mask_blured.to(imgout.device)
    imgout = imgout * mask_blured  # Apply the blurred mask to the output image to get the final membrane profile
    return imgout


if __name__ == "__main__":
    fpath_img = r"/home/astar/Projects/vesicles_data/mackinnon_03182023/misc/mackinnon_03182023_mrc/000022314839760130216_VSM-71-3_815_005_Nov04_04.01.02_X+1Y-1-1_patch_aligned_doseweighted.mrc"

    fpath_mask = r"/home/astar/Projects/vesicles_data/mackinnon_03182023/misc/labels/000022314839760130216_VSM-71-3_815_005_Nov04_04.01.02_X+1Y-1-1_patch_aligned_doseweighted.png"

    img, header, voxel_size = load_mrc(fpath_img)
    parameters = read_parameters_from_yaml_file()  # Read parameters from the YAML file
    img = img.astype(np.float64)
    img = downsample_micrograph(img,voxel_size[0], parameters["r"], "center")
    mask = read_img(fpath_mask, mask=True)  # Read the segmentation mask

    img_tensor, mask_blured, row_idx, col_idx = prepare_micrograph(img, mask, parameters, parameters["r"])
    #row_idx = row_idx[[6197,]]
    #col_idx = col_idx[[6197,]]


    membrane_blend = get_membrane_blending(img_tensor, mask_blured, row_idx, col_idx, parameters["r"],parameters["w"])
    membrane_basis = fit_membrane(img_tensor,mask_blured,row_idx,col_idx,parameters)

    membrane_blend = membrane_blend.to("cpu").numpy()  # Move the membrane tensor to CPU and convert to NumPy array for visualization
    sub_basis = img - membrane_basis # Subtract the membrane from the original image to get the image with subtracted membranes
    sub_blend = img - membrane_blend # Subtract the membrane from the original image to get the image with subtracted membranes
    r = parameters["r"]
    dog_metric_blend,_,_ = subtraction_metric(img[3*r:-3*r,3*r:-3*r], sub_blend[3*r:-3*r,3*r:-3*r], mask[3*r:-3*r,3*r:-3*r])
    dog_metric_basis,_,_ = subtraction_metric(img[3*r:-3*r,3*r:-3*r], sub_basis[3*r:-3*r,3*r:-3*r], mask[3*r:-3*r,3*r:-3*r])
    print(f"Subtraction metric for blend method: {dog_metric_blend}")
    print(f"Subtraction metric for basis method: {dog_metric_basis}")
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(membrane_basis, cmap='gray')
    plt.title('membrane basis')
    plt.grid(False)
    plt.subplot(1,2,2)
    plt.imshow(membrane_blend, cmap='gray')
    plt.title('membrane blend')
    plt.grid(False)



    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(sub_blend, cmap='gray')
    plt.title('subtraction blend')
    plt.grid(False)
    plt.subplot(1,3,2)
    plt.imshow(sub_basis, cmap='gray')
    plt.title('subtraction basis')
    plt.grid(False)
    plt.subplot(1,3,3)
    plt.imshow(img, cmap='gray')
    plt.title('original image')
    plt.grid(False)


    plt.figure()
    plt.imshow((membrane_blend - membrane_basis)[3*r:-3*r,3*r:-3*r], cmap='gray')
    plt.title("membrane difference")
    plt.grid(False)

    plt.figure()
    plt.plot(np.mean(img[356:364,630:670],axis=0), label='original')
    plt.plot(np.mean(membrane_blend[356:364,630:670],axis=0), label='membrane blend')
    plt.plot(np.mean(membrane_basis[356:364,630:670],axis=0), label='membrane basis')
    plt.legend()
    plt.title("Membrane profiles along a line")
    plt.grid(False)
    plt.show()
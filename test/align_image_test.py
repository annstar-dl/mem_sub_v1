import torch
from  sampling_grid import get_sampling_grid, select_points_within_boundary
from align_image import rotate_images_kornia, align_single_patch_multires
import math
import scipy
from basis_fn import get_radius_of_inner_circle, get_w_function, create_gaussian_disc
from matplotlib import pyplot as plt
from recon_patch import recon_patch, recon_mult_patches

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

def visualize_im(im, title="Image", add_figure=True):
    """
    Visualize the image tensor.
    Args:
        im (torch.Tensor): Image tensor of shape (C, H, W).
    """
    if add_figure:
        plt.figure()
    if im.dim()>2:
        im = im.permute(1, 2, 0)  # Change shape to (H, W, C) for visualization
        plt.imshow(im)
    else:
        plt.imshow(im, cmap='gray')  # For single channel images
    plt.axis('off')
    plt.title(title)

def visualize_2_images(im1, im2, title1="Image 1", title2="Image 2"):
    """
    Visualize two images side by side.
    Args:
        im1 (torch.Tensor): First image tensor of shape (C, H, W).
        im2 (torch.Tensor): Second image tensor of shape (C, H, W).
        title1 (str): Title for the first image.
        title2 (str): Title for the second image.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    visualize_im(im1, title=title1, add_figure=False)
    plt.subplot(1, 2, 2)
    visualize_im(im2, title=title2, add_figure=False)


def visualize_3_images(im1, im2, im3, title1="Image 1", title2="Image 2", title3="Image 3"):
    """
    Visualize three images side by side.
    Args:
        im1 (torch.Tensor): First image tensor of shape (C, H, W).
        im2 (torch.Tensor): Second image tensor of shape (C, H, W).
        im3 (torch.Tensor): Third image tensor of shape (C, H, W).
        title1 (str): Title for the first image.
        title2 (str): Title for the second image.
        title3 (str): Title for the third image.
    """
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    visualize_im(im1, title=title1, add_figure=False)
    plt.subplot(1, 3, 2)
    visualize_im(im2, title=title2, add_figure=False)
    plt.subplot(1, 3, 3)
    visualize_im(im3, title=title3, add_figure=False)


def extract_small_patch_with_padding(img, x, y, r):
    """
    Extract a small patch from the image. Image is padded in case the patch goes out of bounds.

    Args:
        img (torch.Tensor): The image tensor of shape (C, H, W).
        x (int): X coordinate of the center of the patch.
        y (int): Y coordinate of the center of the patch.
        r (int): Radius of the patch to extract.

    Returns:
        torch.Tensor: The extracted patch tensor of shape (C, 2*r+1, 2*r+1).
    """
    img = img.unsqueeze(0) if img.dim() == 3 else img  # Ensure img is 4D tensor (C, H, W)
    img_padded = torch.nn.functional.pad(img, (r, r, r, r), mode='constant', value=0)  # Pad the image
    # Extract the patch from the padded image
    patch = img_padded[:,:, y:y+2*r, x:x+2*r]  # Extract the patch
    return patch.squeeze(0)




def select_one_patch(img, row_idx,col_idx, r, indx):
    return img[row_idx[indx]-r:row_idx[indx]+r+1, col_idx[indx]-r:col_idx[indx]+r+1]

def visualize_rotations(imgs, angles):
    plt.figure()
    fig, axes = plt.subplots(nrows=2, ncols=len(imgs)//2)
    for i in range(2):
        for j in range(len(imgs)//2):
            axes[i,j].imshow(imgs[i*j+j].squeeze(), cmap='gray')
            axes[i,j].axis('off')
            axes[i,j].set_title(f"Angle: {angles[i*j+j]:.2f}°")

def visualize_rotations_1d(imgs, angles, prof, losses):
    plt.figure()
    fig, axes = plt.subplots(2,len(imgs))
    for i in range(len(imgs)):

        axes[0,i].imshow(imgs[i].squeeze(), cmap='gray')
        axes[0,i].axis('off')
        axes[0,i].set_title(f"Angle: {angles[i]:.2f}°")
        axes[1,i].imshow(prof[i].squeeze().cpu().numpy())
        axes[1,i].set_title(f"{angles[i]:.2f}°, loss {losses[i]:.2f}")


def syn_image(n):
    img = torch.zeros((n, n), dtype=torch.float64)
    img[n//2 -1 :n//2+1,:] = 1.0  # Set the center pixel to 1
    return img

def sum_prof(img):

    return torch.sum(img, dim=3)  # Sum the rotated images along the channel dimension and normalize by the number of angles

def calculate_mse_loss(img, prof):

    prof = prof.unsqueeze(3).expand(-1, -1, -1, img.shape[3])  # Expand the profile into an image for each angle
    return torch.sum((prof-img)**2, dim=(1, 2, 3)), prof

def vizualize_img_pointer(img,row_idx, col_idx):
    """
    Visualize the image tensor.
    Args:
        img (torch.Tensor): Image tensor of shape (C, H, W).
    """
    img[row_idx-3:row_idx+3, col_idx-3:col_idx+3] = 255  # Set the center pixel to 1
    plt.figure()
    plt.imshow(img, cmap="gray")


def align_image_test():
    r = 20
    r_in = int(r / math.sqrt(2.0)-1) # Radius of the inner circle
    idx = 0
    cntr = r
    w = get_w_function(r_in)  # Get the weights for the Gaussian kernel
    file_path = r'/home/astar/Projects/data_from_matlab_code/mk_1.mat'
    img = load_image_from_mat(file_path, "img")
    mask = load_image_from_mat(file_path, "mask")
    img = torch.tensor(img, dtype=torch.float64)
    mask = torch.tensor(mask, dtype=torch.float32)
    img = img - torch.mean(img)  # Center the patch around zero
    file_path = r"/home/astar/Projects/data_from_matlab_code/mk_1_coord_theta_basis.mat"
    angle_matlab = load_image_from_mat(file_path, "theta")
    basis_matlab = load_image_from_mat(file_path, "basis")
    mask, row_idx, col_idx = get_sampling_grid(mask, 4, 4)  # Get the sampling grid from the mask

    row_idx = load_image_from_mat(file_path, "x")-1
    row_idx = torch.tensor(row_idx).to(torch.int64)
    col_idx = load_image_from_mat(file_path, "y")-1
    col_idx = torch.tensor(col_idx).to(torch.int64)
    patch = select_one_patch(img,row_idx,col_idx,r,0)
    visualize_im(patch,"Original Patch")  # Visualize the original patch
    #patch = syn_image(patch.shape[-1])
    angle_range = torch.arange(-80,80,20)

    patch_for_rot = patch.unsqueeze(0).unsqueeze(0).expand(len(angle_range), -1, -1, -1).contiguous()
    rot_imgs = rotate_images_kornia(patch_for_rot,angle_range.to(torch.float32))
    rot_imgs = rot_imgs[...,cntr-r_in:cntr+r_in+1, cntr-r_in:cntr+r_in+1]  # Crop the images to the neighbourhood size
    w = get_w_function(rot_imgs.shape[3] // 2)
    w_prof = torch.sum(w, dim=1)
    print("w prof",w_prof)
    prof = sum_prof(rot_imgs *w)# Rotate the image by the angles
    plt.plot(w[0])
    plt.title("Weights for Gaussian kernel")
    losses, prof = calculate_mse_loss(rot_imgs, prof)  # Calculate the MSE loss between the rotated images and the profile
    angle_tmp = angle_range[torch.argmin(losses)]
    visualize_rotations_1d(rot_imgs,angle_range,prof,losses)

    angle_my = align_single_patch_multires(patch, cntr, r_in, w, -90, 90, 1)
    print(fr"Angle my {angle_my}, angle tmp {angle_tmp}")
    #angle = align_single_patch(patch, cntr, r_in, w, -90., 90.0, 1.0)  # Align the image using the center and radius
    #print(f"Angle python {angle}, angle matlab {angles_matlab[idx]}")
    binaryImage, gaussWt = create_gaussian_disc(2 * [(2 * r_in + 1)], r_in)  # Create a binary disc and Gaussian weights
    basis = recon_patch(patch, cntr, r_in, w, gaussWt,
                               angle_my)  # Reconstruct the patch using the basis functions
    visualize_3_images(patch, basis, torch.tensor(basis_matlab), "Org","Reconstructed Patch","basis_matlab")

if __name__== "__main__":
    align_image_test()
    plt.show()
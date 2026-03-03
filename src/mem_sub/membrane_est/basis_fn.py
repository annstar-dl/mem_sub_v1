import math
from mem_sub.membrane_est.align_image import align_single_patch, align_multiple_patches
from mem_sub.membrane_est.recon_patch import recon_patch, recon_mult_patches
from mem_sub.membrane_est.sub_utils import get_patches_from_image_adv_indexing
import torch


def create_gaussian_disc(im_size, radius):
    """
    Create a binary disc using Gaussian kernel.
    Args:
        im_size (int,int): Size of the image (height and width).
        radius (int): Radius of the disc.
    Returns:
        torch.Tensor: Binary disc of shape (im_size, im_size).
        torch.Tensor: Gaussian weights for the disc of shape (im_size, im_size).
    """
    if isinstance(im_size, int):
        raise ValueError("im_size should be a tuple of (height, width), not a single integer.")
    rows = im_size[0]
    cols = im_size[1]
    y, x = torch.meshgrid(torch.arange(cols), torch.arange(rows))
    y = y.to(torch.float64)  # Ensure y is in double precision
    x = x.to(torch.float64)  # Ensure x is in double precision
    centerX = cols /2 - 0.5 # Center X coordinate
    centerY = rows /2 - 0.5 # Center Y coordinate
    centerX = torch.tensor(centerX, dtype=torch.float64)  # Center X coordinate
    centerY = torch.tensor(centerY, dtype=torch.float64)  # Center Y coordinate
    #compute radius of each pixel from the center
    r = torch.sqrt(((x - centerX) ** 2 + (y - centerY) ** 2))
    # Create a Gaussian disc using the radius
    binaryImage = r <= radius # Create a binary disc
    sigma = radius / 2.5  # Standard deviation for Gaussian kernel
    # Create Gaussian weights based on the distance from the center
    gaussWt = torch.exp(-r**2 / (2 * sigma ** 2))
    edge_val = gaussWt[int(centerX),-1] # Get the value at the edge of the disc
    gaussWt = torch.max(gaussWt-edge_val,torch.tensor(0.0))  # Ensure the maximum value is at the edge
    gaussWt = gaussWt * binaryImage  # Apply the binary mask to the Gaussian weights
    gaussWt = gaussWt / torch.sum(gaussWt)  # Normalize the Gaussian weights
    return binaryImage, gaussWt

def get_w_function(r):
    """Get the weights for the Gaussian kernel.
    Arg:
    r (int): Radius of the inner circle inside the neighborhood."""

    # w function for high order approximation
    u = torch.arange(-r, r+1, dtype=torch.float64) / r
    w = 1.40625 - 4.6875 * u ** 2 + 3.28125 * u ** 4
    w = w / torch.sum(w)  # Normalize the weights
    w = w.repeat(2 * r+1, 1)  # Repeat to create a 2D weight matrix
    return w

def get_radius_of_inner_circle(r):
    """
    Get the radius of the inner circle based on the neighbourhood radius.

    Args:
        r (int): Outer radius of the neighbourhood.

    Returns:
        int: Radius of the inner circle.
    """
    return math.floor(r / math.sqrt(2.0)) - 1  # Radius of the inner circle

def get_basis_sequential(dataimg,row_idx,col_idx,r):
    """
    Get bases from the image data patch by patch. Basis is a membrane profile at a point.

    Args:
        img (torch.Tensor): Input image of shape (H,W), where N is the number of samples
                            and H, W are the height and width of the image.
        dataimg (torch.Tensor): The image from previous processing step of shape (H,W).
        mask (torch.Tensor): Segmentation mask of shape (H,W) to apply on the image.
        row_idx (torch.Tensor): X coordinates of grid of shape (N), number of grid points.
        col_idx (torch.Tensor): Y coordinates of grid of shape (N), number of grid points.
        r (int): Radius of a neighbourhood.

    Returns:
        tuple: (torch.Tensor, torch.Tensor)
        1.torch.Tensor: Bases(Reconstructed images of neighbourhoods around grid point)
         of shape (N,r,r), where N is the number of bases and r is inner radius of neighbourhood
         around sampling grid point.
        2.torch.Tensor: The angles of the membrane profiles at each grid point of shape (N,).
    """
    # Check if the input image is 2D or 3D
    if dataimg.dim() != 2:
        raise ValueError("Input image must be a 2D tensor, got {} dimensions".format(dataimg.dim()))
    cntr = r

    r_in = get_radius_of_inner_circle(r)
    w = get_w_function(r_in)  # Get the weights for the Gaussian kernel

    binaryImage, gaussWt = create_gaussian_disc(2*[(2*r_in+1)], r_in)  # Create a binary disc and Gaussian weights
    nGrid = row_idx.shape[0]  # Number of grid points
    basis = torch.zeros((nGrid, w.shape[0], w.shape[1]), dtype=torch.float64)  # Initialize basis tensor
    thetas = torch.zeros((nGrid,), dtype=torch.float64)  # Initialize theta tensor for angles
    for i in range(nGrid):
        img1 = dataimg[row_idx[i]-r:row_idx[i]+r+1, col_idx[i]-r:col_idx[i]+r+1]  # Extract the neighborhood around the grid point
        if img1.shape[0] != 2 * r + 1 or img1.shape[1] != 2 * r + 1:
            raise ValueError(f"Extracted patch size {img1.shape} does not match expected size {(2 * r + 1, 2 * r + 1)}"
                             f" for grid point ({row_idx[i]}, {col_idx[i]}). ")
        theta = align_single_patch(img1, cntr, r_in,w,-90.,90.0,1.0)  # Align the image using the center and radius
        patchImg = recon_patch(img1, cntr, r_in, w, gaussWt, theta)  # Calculate the basis functions for the patch
        basis[i] = patchImg  # Store the reconstructed patch in the basis tensor
        thetas[i] = theta  # Store the angle for the current grid point
    return basis, thetas



def get_basis(dataimg,row_idx,col_idx,r):
    """
    Get bases from the image data multiple patches at once. Basis is a membrane profile at a point.

    Args:
        dataimg (torch.Tensor): The image from previous processing step of shape (H,W).
        row_idx (torch.Tensor): X coordinates of grid of shape (N), number of grid points.
        col_idx (torch.Tensor): Y coordinates of grid of shape (N), number of grid points.
        r (int): Radius of a neighbourhood.

    Returns:
        tuple: (torch.Tensor, torch.Tensor)
        1.torch.Tensor: Bases(Reconstructed images of neighbourhoods around grid point)
         of shape (N,r,r), where N is the number of bases and r is inner radius of neighbourhood
         around sampling grid point.
        2.torch.Tensor: The angles of the membrane profiles at each grid point of shape (N,).
    """
    # Check if the input image is 2D or 3D
    if dataimg.dim() != 2:
        raise ValueError("Input image must be a 2D tensor, got {} dimensions".format(dataimg.dim()))
    cntr = r
    r_in = get_radius_of_inner_circle(r)
    w = get_w_function(r_in)  # Get the weights for the Gaussian kernel

    binaryImage, gaussWt = create_gaussian_disc(2*[(2*r_in+1)], r_in)  # Create a binary disc and Gaussian weights
    imgs_subset = get_patches_from_image_adv_indexing(dataimg, r, row_idx, col_idx)  # Get patches from the image using the specified radius
    imgs_subset = imgs_subset.unsqueeze(1)  # Add channel dimension
    # Move imgs_subset to GPU if available
    if torch.cuda.is_available():
        imgs_subset = imgs_subset.to("cuda")
        w = w.to("cuda")  # Move weights to GPU
        gaussWt = gaussWt.to("cuda")
    theta = align_multiple_patches(imgs_subset,cntr, r_in,w,-90.,90.0,1.0)  # Align the image using the center and radius
    basis = recon_mult_patches(imgs_subset, cntr, r_in, w, gaussWt, theta)  # Reconstruct the patch using the basis functions
    return basis, theta




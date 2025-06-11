import math
from align_image import align_image_all, align_single_patch
from recon_patch import recon_patch
import torch


def create_gaussian_disc(im_size, r):
    """
    Create a binary disc using Gaussian kernel.
    Args:
        im_size (int): Size of the image (height and width).
        r (int): Radius of the disc.
    Returns:
        torch.Tensor: Binary disc of shape (im_size, im_size).

    """
    y, x = torch.meshgrid(torch.arange(im_size), torch.arange(im_size))
    center = im_size // 2
    gaussWt = torch.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * r ** 2))
    binary_disc = (gaussWt > 0.5).float()  # Convert to binary disc
    return binary_disc, gaussWt

def get_w_function(r):
    """Get the weights for the Gaussian kernel.
    Arg:
    r (int): Radius of the inner circle inside the neighborhood."""

    # w function for high order approximation
    u = torch.arange(-r, r+1, dtype=torch.float32) / r
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

def get_basis(img,dataimg,mask,x,y,r):
    """
    Get bases from the image data using least squares method.

    Args:
        img (torch.Tensor): Input image of shape (H,W), where N is the number of samples
                            and H, W are the height and width of the image.
        dataimg (torch.Tensor): The image from previous processing step of shape (H,W).
        mask (torch.Tensor): Segmentation mask of shape (H,W) to apply on the image.
        x (torch.Tensor): X coordinates of grid of shape (N), number of grid points.
        y (torch.Tensor): Y coordinates of grid of shape (N), number of grid points.
        r (int): Radius of a neighbourhood.

    Returns:
        torch.Tensor: Bases(Reconstructed images of neighbourhoods around grid point)
         of shape (D,r,r), where N is the number of bases and D is the dimension.
    """
    cntr = r+1
    imgout = torch.zeros_like(img)
    wtimg = torch.zeros_like(img)

    r_in = get_radius_of_inner_circle(r)
    w = get_w_function(r_in)  # Get the weights for the Gaussian kernel

    binaryImage, gaussWt = create_gaussian_disc(2*r_in+1, r_in)  # Create a binary disc and Gaussian weights
    nGrid = x.shape[0]  # Number of grid points
    basis = torch.zeros((nGrid, w.shape[0], w.shape[1]), dtype=torch.float32)  # Initialize bases tensor
    for i in range(nGrid):
        xi = x[i]
        yi = y[i]
        img1 = dataimg[yi-r:yi+r+1, xi-r:xi+r+1]  # Extract the neighborhood around the grid point
        theta = align_single_patch(img1, cntr, r_in,w,-90.,90.0,1.0)  # Align the image using the center and radius
        patchImg = recon_patch(img1, cntr, r_in, w, gaussWt, theta)  # Reconstruct the patch using the basis functions
        basis[i] = patchImg  # Store the reconstructed patch in the bases tensor
        imgout[yi-r_in:yi+r_in+1, xi-r_in:xi+r_in+1] += patchImg
        wtimg[yi-r_in:yi+r_in+1, xi-r_in:xi+r_in+1] += gaussWt
    imgout = imgout / wtimg  # Normalize the output image by the weights
    return basis

def fit_bases_to_data(data, bases, x,y,r):
    """
    Fit bases to data using least squares method.

    Args:
        data (torch.Tensor): Input image of shape (N,H,W), where N is the number of samples
                             and H, W are the height and width of the image.
        bases (torch.Tensor): Bases of shape (M, D), where M is the number of bases.
        weights (torch.Tensor, optional): Weights for each sample of shape (N,). If None, all samples are equally weighted.

    Returns:
        torch.Tensor: Fitted coefficients of shape (M,).
    """

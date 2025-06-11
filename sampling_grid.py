import torch
from skimage.morphology import disk, dilation
from skimage.filters import gaussian
import kornia

def dilate_mask_skimage(mask, d):
    """
    Dilate the mask using a disk-shaped structuring element.

    Args:
        mask (torch.Tensor): Segmentation mask of shape (H,W) to apply on the image.
        d (int): amount by which the mask is dilated.

    Returns:
        torch.Tensor: dilated mask of shape (H,W) after dilation.
    """
    dilated_mask = mask.cpu().numpy()  # Convert to numpy array for skimage processing
    se = disk(1)  # Create a disk-shaped structuring element
    for _ in range(d):
        dilated_mask = dilation(dilated_mask, se)
    dilated_mask = torch.tensor(dilated_mask, dtype=mask.dtype, device=mask.device)  # Convert back to tensor
    return dilated_mask

def gaussian_filter_skimage(mask, sigma):
    """
    Apply Gaussian filter to the mask using skimage.

    Args:
        mask (torch.Tensor): Segmentation mask of shape (H,W) to apply on the image.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        torch.Tensor: Mask after applying Gaussian filter.
    """

    filtered_mask = mask.cpu().numpy()  # Convert to numpy array for skimage processing
    filtered_mask = gaussian(filtered_mask, sigma=sigma, mode='reflect', preserve_range=True)
    filtered_mask = torch.tensor(filtered_mask, dtype=mask.dtype, device=mask.device)  # Convert back to tensor
    return filtered_mask

def dilate_mask(mask, d):
    """
    Dialate the mask using a disk-shaped structuring element.

    Args:
        mask (torch.Tensor): Segmentation mask of shape (H,W) to apply on the image.
        d (int): amount by which the mask is dilated.

    Returns:
        torch.Tensor: dilated mask of shape (H,W) after dilation.
    """
    dilated_mask = mask.clone()  # Clone the original mask to avoid modifying it
    for _ in range(d):
        # Use a disk-shaped structuring element for dilation
        # The disk function creates a binary disk with radius d
        # We use kornia's dilation function to apply the dilation operation
        dilated_mask = gaussian_filter(dilated_mask, kernel_size=(3,3), sigma=(0.5, 0.5))
        dilated_mask = dilated_mask>0.5  # Threshold the mask to create a binary mask
        dilated_mask = dilated_mask.float()

    return dilated_mask

def gaussian_filter(mask, kernel_size, sigma):
    """
    Apply Gaussian filter to the mask.

    Args:
        mask (torch.Tensor): Segmentation mask of shape (H,W) to apply on the image.
        kernel_size (torch.Tensor): Gaussian kernel to apply on the mask of shape (1,1)
        sigma (torch.Tensor): Standard deviation for Gaussian kernel of shape (1,1)

    Returns:
        torch.Tensor: Mask after applying Gaussian filter.
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # Ensure the mask is a 4D tensor (C, H, W)
    elif mask.dim() == 3:
        mask = mask.unsqueeze()  # Ensure the mask is a 4D tensor
    elif mask.dim() != 4:
        raise ValueError(f"Expected mask to be 2D or 3D tensor, got {mask.dim()} dimensions")
    # Use kornia's Gaussian blur function to apply the filter
    filtered_mask = kornia.filters.gaussian_blur2d(mask, kernel_size, sigma)
    filtered_mask = filtered_mask.squeeze(0).squeeze(0)  # Remove the extra dimensions
    return filtered_mask

def select_points_within_boundary(image,r, x_indices, y_indices):
    """
    Select points within the boundary of the image based on the radius.

    Args:
        image (tensor): Size of the image (assumed square).
        r (int): Radius to define the boundary.
        x_indices (torch.Tensor): X coordinates of grid points.
        y_indices (torch.Tensor): Y coordinates of grid points.

    Returns:
        torch.Tensor: Filtered x coordinates within the boundary.
        torch.Tensor: Filtered y coordinates within the boundary.
    """
    image_size_x, image_size_y = image.shape[-1], image.shape[-2]  # Assuming square image
    mask = ((x_indices >= r) & (x_indices < image_size_x - r) &
            (y_indices >= r) & (y_indices < image_size_y - r))
    x_indices = x_indices[mask]  # Filter x indices within the boundary
    y_indices = y_indices[mask]  # Filter y indices within the boundary
    #sort the indices by x
    sorted_indices = torch.argsort(x_indices)  # Sort indices based on x coordinates
    x_indices = x_indices[sorted_indices]  # Sort x indices
    y_indices = y_indices[sorted_indices]  # Sort y indices according to sorted x indices
    # sort the indices by y
    sorted_indices = torch.argsort(y_indices)  # Sort indices based on y coordinates
    x_indices = x_indices[sorted_indices]  # Sort x indices
    y_indices = y_indices[sorted_indices]  # Sort y indices according to sorted y indices
    return x_indices, y_indices

def get_sampling_grid(mask, d, w):
    """
    Get sampling grid from the image data using least squares method.

    Args:
        mask (torch.Tensor): Segmentation mask of shape (H,W) to apply on the image.
        d (int): amount by which the mask is dilated.
        w (int): sampling grid step size.(spacing between grid points)

    Returns:
        torch.Tensor: dilated mask of shape (H,W) after dilation.
        torch.Tensor: x coordinates Sampling grid of shape (N,), where N is the number of grid points.
        torch.Tensor: y coordinates Sampling grid of shape (N,), where N is the number of grid points.
    """

    mask = dilate_mask(mask, d)  # Dialate the mask
    # Create a grid of points based on the mask
    grid = torch.zeros_like(mask, dtype=torch.float32)  # Initialize grid
    grid[::w, ::w] = 1  # Set grid points at intervals of w
    grid = grid * mask  # Apply the mask to the grid
    # Get the indices of the grid points
    y_indices, x_indices = torch.where(grid > 0)  # Get indices of non-zero elements in the mask
    # Select points within the boundary of the image based on the radius
    #x_indices, y_indices = select_points_within_boundary(mask.shape[0], d, x_indices, y_indices)  # Filter points within the boundary
    ### WHY DILATION IS NEEDED again?
    mask = dilate_mask(mask, 3)
    #mask = gaussian_filter_skimage(mask,2.0)# Dilated the mask again to ensure grid points are within the mask
    mask = gaussian_filter(mask, kernel_size=(3, 3), sigma=(2.0, 2.0))  # Apply Gaussian filter to smooth the mask
    mask = mask/ mask.max()  # Normalize the mask to [0, 1] range
    return mask , x_indices, y_indices# Return the grid and coordinates as float tensors
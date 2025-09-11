import torch
from skimage.morphology import disk, dilation
from skimage.filters import gaussian
import kornia
import torch.nn.functional as F

def creat_idx_batches_for_parl_sum(row_idx, col_idx, r, step):

    init_rows = []
    init_cols = []
    nb_batches_1d = (2*r+1)//step+(1 if (2*r+1)%step!=0 else 0) # Number of batches in one dimension
    for i in range(nb_batches_1d):
        for j in range(nb_batches_1d):
            init_rows.append(min(row_idx)+ i * step)
            init_cols.append(min(col_idx) + j * step)

    batched_row_idxs = []
    batched_col_idxs = []
    bases_idxs = []
    for init_row,init_col in zip(init_rows, init_cols):
        batch_row_idxs = []
        batch_col_idxs = []
        basis_idxs = []
        for i in range(len(row_idx)):
            rr = row_idx[i]
            c = col_idx[i]
            dev = step*((2*r+1)//step + (1 if (2*r+1)%step!=0 else 0))  # Calculate the deviation for the grid
            if (rr -init_row) % dev== 0 and (c-init_col) % dev== 0:
                batch_row_idxs.append(rr)
                batch_col_idxs.append(c)
                basis_idxs.append(i)
            elif rr== init_row and c == init_col:  # If the point is exactly at the initial point
                batch_row_idxs.append(rr)
                batch_col_idxs.append(c)
                basis_idxs.append(i)
        batched_row_idxs.append(torch.tensor(batch_row_idxs, dtype=torch.int64))
        batched_col_idxs.append(torch.tensor(batch_col_idxs, dtype=torch.int64))
        bases_idxs.append(torch.tensor(basis_idxs, dtype=torch.int64))
    return batched_row_idxs, batched_col_idxs, bases_idxs

def sum_patches_back_batched(img, patches, row_idxs, col_idxs,bases_idxs, r):

    img_new = torch.zeros_like(img)  # Create a copy of the image to avoid modifying the original
    normalizer = torch.zeros_like(img)  # Create a normalizer to avoid double counting
    row_grid, col_grid = torch.meshgrid([torch.arange(-r, r + 1), torch.arange(-r, r + 1)])

    for b_idx in range(len(row_idxs)):
        row_grid_expand = row_grid.unsqueeze(0).expand(len(row_idxs[b_idx]), -1, -1)
        col_grid_expand = col_grid.unsqueeze(0).expand(len(row_idxs[b_idx]), -1, -1)
        row_grid_batched = row_grid_expand + row_idxs[b_idx].unsqueeze(1).unsqueeze(2)
        col_grid_batched = col_grid_expand + col_idxs[b_idx].unsqueeze(1).unsqueeze(2)

        img_new[row_grid_batched, col_grid_batched] += patches[bases_idxs[b_idx]]
        normalizer[row_grid_batched, col_grid_batched] += 1  # Increment the normalizer at the patch location
    img_new[normalizer != 0]  /= normalizer[normalizer != 0] # Normalize the summed image to avoid double counting
    return img_new, normalizer

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

def create_disk_kernel(radius: int) -> torch.Tensor:
    """Create a disk-shaped binary kernel as a tensor."""
    diameter = 2 * radius + 1
    y, x = torch.meshgrid(torch.arange(diameter), torch.arange(diameter), indexing='ij')
    center = radius
    dist = ((x - center)**2 + (y - center)**2).float().sqrt()
    disk = (dist <= radius).float()
    return disk.view(1, 1, diameter, diameter)



def dilate_mask(mask: torch.Tensor,radius=3 ) -> torch.Tensor:
    """
    Simulates imdilate(mask, se) using 2D convolution.

    Args:
        mask: (B, 1, H, W) binary image tensor (0s and 1s)
        se_kernel: (1, 1, kH, kW) binary structuring element (disk or square)

    Returns:
        dilated: (B, 1, H, W) binary dilated mask
    """
    se_kernel = create_disk_kernel(radius=radius)  # Create a disk-shaped kernel with radius 1
    # Pad so output size matches input size
    padding = se_kernel.shape[-1] // 2

    # Perform binary dilation: convolve and threshold
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # Ensure mask is a 4D tensor (B, C, H, W)
    out = F.conv2d(mask.to(torch.float32), se_kernel, padding=padding)
    dilated = (out > 0).float()  # if any neighbor is 1, output is 1
    return dilated.squeeze()

def dilate_mask_kornia(mask,kernel_size=(3,3), sigma=(0.5,0.5)):
    """
    Dilate the mask using a disk-shaped structuring element.

    Args:
        mask (torch.Tensor): Segmentation mask of shape (H,W) to apply on the image.
        kernel_size (torch.Tensor): Gaussian kernel to apply on the mask of shape (1,1)
        sigma (torch.Tensor): Standard deviation for Gaussian kernel of shape (1,1)

    Returns:
        torch.Tensor: dilated mask of shape (H,W) after dilation.
    """
    dilated_mask = mask.clone()  # Clone the original mask to avoid modifying it
    dilated_mask = gaussian_filter(dilated_mask, kernel_size=kernel_size, sigma=sigma)
    dilated_mask = dilated_mask>0.5  # Threshold the mask to create a binary mask
    # Perform binary dilation: convolve and threshold
    dilated_mask = dilated_mask.to(torch.float64)  # Ensure the mask is in float32 format
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
    filtered_mask = mask.clone()  # Clone the original mask to avoid modifying it
    if filtered_mask.dim() == 2:
        filtered_mask = filtered_mask[None,None,:,:] # Ensure the mask is a 4D tensor (C, H, W)
    elif filtered_mask.dim() == 3:
        filtered_mask = filtered_mask.unsqueeze()  # Ensure the mask is a 4D tensor
    elif filtered_mask.dim() != 4:
        raise ValueError(f"Expected mask to be 2D or 3D tensor, got {mask.dim()} dimensions")
    # Use kornia's Gaussian blur function to apply the filter
    filtered_mask = kornia.filters.gaussian_blur2d(filtered_mask, kernel_size, sigma)
    filtered_mask = filtered_mask.squeeze() # Remove the extra dimensions
    return filtered_mask

def select_points_within_boundary(image, r, row_idxs, col_idxs):
    """
    Select points within the boundary of the image based on the radius.
    Then sort the indices by x and y coordinates.
    Args:
        image (tensor): Size of the image (assumed square).
        r (int): Radius to define the boundary.
        row_idxs (torch.Tensor): X coordinates of grid points.
        col_idxs (torch.Tensor): Y coordinates of grid points.

    Returns:
        torch.Tensor: Filtered x coordinates within the boundary.
        torch.Tensor: Filtered y coordinates within the boundary.
    """
    row_size, col_size = image.shape[-2], image.shape[-1]  # Assuming square image
    mask = ((row_idxs >= r) & (row_idxs < row_size - r) &
            (col_idxs >= r) & (col_idxs < col_size - r))
    row_idxs = row_idxs[mask]  # Filter row indices within the boundary
    col_idxs = col_idxs[mask]  # Filter col indices within the boundary
    #sort the indices by x
    return row_idxs, col_idxs

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
    mask = dilate_mask(mask, 1)  # Dialate the mask

    for _ in range(d):
        mask = dilate_mask(mask, 1)
    # Create a grid of points based on the mask
    grid = torch.zeros_like(mask, dtype=torch.uint8)  # Initialize grid
    grid[::w, ::w] = 1  # Set grid points at intervals of w
    grid = grid * mask  # Apply the mask to the grid
    # Get the indices of the grid points
    row_indices, col_indices = torch.where(grid > 0)  # Get indices of non-zero elements in the mask
    # Select points within the boundary of the image based on the radius
    for _ in range(3):
        mask = dilate_mask(mask, 1)
    mask = gaussian_filter(mask, kernel_size=(11, 11), sigma=(3.5, 3.5))  # Apply Gaussian filter to smooth the mask
    mask = mask/ mask.max()  # Normalize the mask to [0, 1] range
    return mask , row_indices, col_indices# Return the grid and coordinates as float tensors
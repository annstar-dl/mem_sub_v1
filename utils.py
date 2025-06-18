import torch


def get_patches_from_image(img, r, row_idxs, col_idxs):
    """
    Get patches from the image using the specified radius.

    Args:
        img (torch.Tensor): Input image of shape (H,W), where H, W are the height and width of the image.
        r (int): Radius of a neighbourhood.

    Returns:
        torch.Tensor: Patches of shape (N, 2*r+1, 2*r+1), where N is the number of patches.
    """
    nGrid = len(row_idxs)  # Number of grid points
    patches = torch.zeros([nGrid,1, 2 * r + 1, 2 * r + 1], dtype=torch.float64)  # Initialize patches tensor)
    for i in range(nGrid):
        patches[i] = img[row_idxs[i] - r:row_idxs[i] + r + 1, col_idxs[i] - r:col_idxs[i] + r + 1].unsqueeze(0)
    return patches

def add_patches_to_image(patches, img, r, row_idxs, col_idxs):
    """
    Get patches from the image using the specified radius.

    Args:
        patches (torch.Tensor): patches of shape (N, 2*r+1, 2*r+1) that need to be pasted into image
        img (torch.Tensor): Input image of shape (H,W), where H, W are the height and width of the image.
        r (int): Radius of a neighbourhood.

    Returns:
        torch.Tensor: Patches of shape (H,W), where N is the number of patches.
    """
    nGrid = len(row_idxs)  # Number of grid points

    for i in range(nGrid):
        img[row_idxs[i] - r:row_idxs[i] + r + 1, col_idxs[i] - r:col_idxs[i] + r + 1] += patches[i]
    return img

def get_patches_from_image_unfold(img, r, row_idxs, col_idxs):
    """
    Get patches from the image using the specified radius.

    Args:
        img (torch.Tensor): Input image of shape (H,W), where H, W are the height and width of the image.
        r (int): Radius of a neighbourhood.

    Returns:
        torch.Tensor: Patches of shape (N, 2*r+1, 2*r+1), where N is the number of patches.
    """
    # Add batch and channel dimensions if missing
    if img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)  # shape: (1,1,H,W)
    elif img.dim() == 3:
        img = img.unsqueeze(0)  # shape: (1,C,H,W)
    else:
        raise ValueError("Input image must be 2D or 3D tensor.")

    k = 2 * r + 1
    patches_all = torch.nn.functional.unfold(img, kernel_size=k, stride=s, padding=0).squeeze(0).T  # (num_patches, k*k)
    H, W = img.shape[-2:]
    # Compute linear indices for the requested patch centers
    idxs = row_idxs * W + col_idxs
    # Get all possible patch centers
    valid_row = torch.arange(r, H - r)
    valid_col = torch.arange(r, W - r)
    grid_row, grid_col = torch.meshgrid(valid_row, valid_col, indexing='ij')
    all_centers = (grid_row.flatten() * W + grid_col.flatten())
    # Map requested indices to their position in the unfolded output
    idx_map = {int(c.item()): i for i, c in enumerate(all_centers)}
    patch_idxs = [idx_map[int(idx.item())] for idx in idxs]
    selected_patches = patches_all[patch_idxs]  # (N, k*k)
    patches = selected_patches.view(-1, 1, k, k)  # (N,1,k,k)
    return patches
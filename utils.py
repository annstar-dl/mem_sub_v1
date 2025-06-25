import torch
import yaml
import os


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

def add_patches_to_image_fold(patches, img, r, s, row_idxs, col_idxs):
    """
    Add patches to the image using the specified radius and stride.
    :param patches:
    :param img:
    :param r:
    :param s:
    :param row_idxs:
    :param col_idxs:
    :return:
    """
    if img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)  # shape: (1,1,H,W)
    elif img.dim() == 3:
        img = img.unsqueeze(0)  # shape: (1,C,H,W)
    else:
        raise ValueError("Input image must be 2D or 3D tensor.")
    k = 2 * r + 1
    unfold = torch.nn.Unfold(kernel_size=k, stride=s, padding=r)  # Create an unfold layer
    patches_all = unfold(img).squeeze(0).T  # (num_patches, k*k)
    H, W = img.shape[-2:]
    # Compute linear indices for the requested patch centers
    idxs = row_idxs * W + col_idxs
    # Get all possible patch centers
    valid_row = torch.arange(0, H,s)
    valid_col = torch.arange(0, W,s)
    grid_row, grid_col = torch.meshgrid(valid_row, valid_col, indexing='ij')
    all_centers = (grid_row.flatten() * W + grid_col.flatten())
    # Map requested indices to their position in the unfolded output
    idx_map = {c.item(): i for i, c in enumerate(all_centers)}
    patch_idxs = [idx_map[idx.item()] for idx in idxs]
    patches_all[:,:] = 0
    patches_all[patch_idxs]=patches.view(-1, k*k)  # (N, k*k)
    # Create a Fold instance with the same parameters and output_size
    fold = torch.nn.Fold(output_size=img.shape[-2:], kernel_size=k, stride=s, padding=r)
    img = fold(patches_all.T.unsqueeze(0))  # (1, C, H, W)
    return img.squeeze()  # Remove the batch dimension, resulting in (H, W)

def get_patches_from_image_unfold(img, r, s, row_idxs, col_idxs):
    """
    Get patches from the image using the specified radius.

    Args:
        img (torch.Tensor): Input image of shape (H,W), where H, W are the height and width of the image.
        r (int): Radius of a neighbourhood.
        s (int): Stride for the patch extraction.
        row_idxs (torch.Tensor): X coordinates of grid points.
        col_idxs (torch.Tensor): Y coordinates of grid points.


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
    unfold = torch.nn.Unfold(kernel_size=k, stride=s, padding=r)  # Create an unfold layer
    patches_all = unfold(img).squeeze(0).T  # (num_patches, k*k)
    H, W = img.shape[-2:]
    # Compute linear indices for the requested patch centers
    idxs = row_idxs * W + col_idxs
    # Get all possible patch centers
    valid_row = torch.arange(0, H,s)
    valid_col = torch.arange(0, W,s)
    grid_row, grid_col = torch.meshgrid(valid_row, valid_col, indexing='ij')
    all_centers = (grid_row.flatten() * W + grid_col.flatten())
    # Map requested indices to their position in the unfolded output
    idx_map = {c.item(): i for i, c in enumerate(all_centers)}
    patch_idxs = [idx_map[idx.item()] for idx in idxs]
    selected_patches = patches_all[patch_idxs]  # (N, k*k)
    patches = selected_patches.view(-1, 1, k, k)  # (N,1,k,k)
    return patches

def read_parameter_from_yaml_file(parameter):
    """
    Read a YAML configuration file and return its contents.

    Args:
        :param parameter:

    Returns:
        value: Contents of the YAML file parameter 
    """
    filename = 'parameters.yml'  # Replace with your YAML file path
    maindir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
    filepath = os.path.join(maindir, filename)  # Construct the full path to the YAML file
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config[parameter]

def read_dict_from_yaml_file():
    """
    Read a YAML configuration file and return its contents as a dictionary.

    Returns:
        dict: Contents of the YAML file.
    """
    filename = 'parameters.yml'  # Replace with your YAML file path
    maindir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
    filepath = os.path.join(maindir, filename)  # Construct the full path to the YAML file
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    pass
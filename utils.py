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

def get_patches_from_image_adv_indexing(img, r, row_idxs, col_idxs):
    """
    Get patches from the image using the specified radius.

    Args:
        img (torch.Tensor): Input image of shape (H,W), where H, W are the height and width of the image.
        r (int): Radius of a neighbourhood.

    Returns:
        torch.Tensor: Patches of shape (N, 2*r+1, 2*r+1), where N is the number of patches.
    """
    row_grid, col_grid = torch.meshgrid([torch.arange(-r, r + 1), torch.arange(-r, r + 1)])
    row_grid = row_grid.unsqueeze(0).expand(len(row_idxs), -1, -1)
    col_grid = col_grid.unsqueeze(0).expand(len(row_idxs), -1, -1)
    row_grid = row_grid + row_idxs.unsqueeze(1).unsqueeze(2)
    col_grid = col_grid + col_idxs.unsqueeze(1).unsqueeze(2)
    patches = img[row_grid, col_grid]  # Extract pixels from the image
    return patches

def creat_idx_batches_for_parl_sum(row_idx, col_idx, r, step):
    """
    Stack the indices of the nonoverlapping patches into batches for parallel summation.
    Args:
        row_idx
        (torch.Tensor): Row indices of the patches.
        col_idx (torch.Tensor): Column indices of the patches.
        r (int): Radius of a neighbourhood.
        step (int): Step size for batching.
    Returns:
        tuple: A tuple containing three lists:
            - batched_row_idxs: List of tensors with row indices for each batch.
            - batched_col_idxs: List of tensors with column indices for each batch.
            - bases_idxs: List of tensors with indices of patches corresponding to each batch.
        """

    init_rows = []
    init_cols = []
    nb_batches_1d = (2*r+1)//step+(1 if (2*r+1)%step!=0 else 0) # Number of batches in one dimension
    nb_batches_1d = (2 * r + 1) // step + (1 if (2 * r + 1) % step != 0 else 0)  # Number of batches in one dimension
    batch_id_init_row_idx = torch.arange(nb_batches_1d) * step + min(row_idx)  # Initial indices for the batches
    batch_id_init_col_idx = torch.arange(nb_batches_1d) * step + min(col_idx)  # Initial indices for the batches
    init_rows, init_cols = torch.meshgrid([batch_id_init_row_idx, batch_id_init_col_idx],
                                          indexing='ij')  # Create a grid of initial points
    init_rows = init_rows.flatten()  # Flatten the grid to get initial row indices
    init_cols = init_cols.flatten()  # Flatten the grid to get initial column indices

    batched_row_idxs = []
    batched_col_idxs = []
    bases_idxs = []
    for init_row, init_col in zip(init_rows, init_cols):
        # Vectorized computation for batch assignment
        rr = row_idx
        cc = col_idx
        dev = step * ((2 * r + 1) // step + (1 if (2 * r + 1) % step != 0 else 0))
        mask = ((rr - init_row) % dev == 0) & ((cc - init_col) % dev == 0)
        mask |= (rr == init_row) & (cc == init_col)
        batch_row_idxs = rr[mask]
        batch_col_idxs = cc[mask]
        basis_idxs = torch.arange(len(row_idx), dtype=torch.int64)[mask]
        batched_row_idxs.append(batch_row_idxs)
        batched_col_idxs.append(batch_col_idxs)
        bases_idxs.append(basis_idxs)

    nb_of_elements_in_batches = [len(batched_row_idxs[i]) for i in
                                 range(len(batched_row_idxs))]  # Number of elements in each batch
    nb_of_elements_in_batches = torch.sum(
        torch.tensor(nb_of_elements_in_batches))  # Total number of elements in all batches
    if nb_of_elements_in_batches != len(row_idx):
        raise ValueError(
            f"Number of elements in batches {nb_of_elements_in_batches} is not equal to number of grid points {len(row_idx)}")

    return batched_row_idxs, batched_col_idxs, bases_idxs

def add_patches_to_image_batched(patches, img, r, row_idxs, col_idxs, bases_idxs):
    """Sum the patches back to the image at specified row and column indices using batched indexing.
    Patches are summed at the specified row and column indices in the image, with one batch containing only non-overlapping patches.
    Args:
        img (torch.Tensor): Input image of shape (H,W), where H, W are the height and width of the image.
        patches (torch.Tensor): Patches tensor of shape (N,2*r+1, 2*r+1).
        row_idxs (list of torch.Tensor): List of row indices for each batch.
        col_idxs (list of torch.Tensor): List of column indices for each batch.
        bases_idxs (list of torch.Tensor): List of indices of patches corresponding to each batch.
        r (int): Radius of a neighbourhood.
    Returns:
        torch.Tensor: Image with patches summed back, of shape (H,W).
        """

    row_grid, col_grid = torch.meshgrid([torch.arange(-r, r + 1), torch.arange(-r, r + 1)])

    for b_idx in range(len(row_idxs)):
        row_grid_expand = row_grid.unsqueeze(0).expand(len(row_idxs[b_idx]), -1, -1)
        col_grid_expand = col_grid.unsqueeze(0).expand(len(row_idxs[b_idx]), -1, -1)
        row_grid_batched = row_grid_expand + row_idxs[b_idx].unsqueeze(1).unsqueeze(2)
        col_grid_batched = col_grid_expand + col_idxs[b_idx].unsqueeze(1).unsqueeze(2)
        img[row_grid_batched, col_grid_batched] += patches[bases_idxs[b_idx]]

    return img

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
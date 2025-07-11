import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import time
from visualization import visualize_2_images, visualize_3_images, visualize_sampling_grid
from sampling_grid import select_points_within_boundary

def vizualize_1_image(image, title="Image"):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)

def init_read_data(nb_images):
    # Initialize the data reading process
    print("Reading MNIST dataset...")
    # Define a transform to convert images to tensors
    transform = transforms.Compose([transforms.Resize((30,30)),transforms.ToTensor()])
    # Download the MNIST test dataset
    mnist_test = datasets.MNIST(root='/home/astar/Projects/VesicleProjection/data', train=True, download=True, transform=transform)
    test_loader = DataLoader(mnist_test, batch_size=nb_images, shuffle=False)
    # Get one batch of 20 images and labels
    images, labels = next(iter(test_loader))
    return images, labels

def extract_grid_centers_from_mask(mask, r, step):
    """
    Extract grid centers from the mask based on the specified radius.

    Args:
        mask (torch.Tensor): Mask tensor of shape (H, W) with 0s and 1s.
        r (int): Radius of the neighborhood.

    Returns:
        torch.Tensor: Row and column indices of the grid centers.
    """
    if mask.dim() != 2:
        raise ValueError("Mask must be a 2D tensor.")

    # Get the indices of the mask where it is 1
    tmp = torch.zeros_like(mask)
    tmp[::step, ::step] = 1  # Downsample the mask to reduce the number of points
    tmp[mask != 1] = 0  # Set the mask points to 1
    row_idx, col_idx = torch.where(tmp == 1)
    row_idx, col_idx = select_points_within_boundary(mask,r, row_idx, col_idx)  # Select points within the boundary of the mask

    return row_idx, col_idx

def extract_patches_from_mask(img, row_idx, col_idx, r):
    """
    Extract only the pixels in the mask from the image.

    Args:
        img (torch.Tensor): Input image tensor of shape (H, W).
        mask (torch.Tensor): Mask tensor of shape (H, W) with 0s and 1s.

    Returns:
        torch.Tensor: Extracted pixels from the image where mask is 1.
    """
    row_grid, col_grid = torch.meshgrid([torch.arange(-r,r+1), torch.arange(-r,r+1)])
    row_grid = row_grid.unsqueeze(0).expand(len(row_idx),-1,-1)
    col_grid = col_grid.unsqueeze(0).expand(len(row_idx), -1, -1)
    row_grid = row_grid + row_idx.unsqueeze(1).unsqueeze(2)
    col_grid = col_grid + col_idx.unsqueeze(1).unsqueeze(2)
    batch_idx = torch.arange(len(row_idx)).unsqueeze(1).unsqueeze(2).expand(-1, row_grid.shape[1], row_grid.shape[2])
    img_batch = img[row_grid, col_grid]  # Extract pixels from the image
    return img_batch
def visualize_patches(img, patches, row_idx, col_idx, r):
    """
    Visualize the extracted patches from the image.

    Args:
        patches (torch.Tensor): Patches tensor of shape (N, 1, 2*r+1, 2*r+1).
        row_idx (torch.Tensor): Row indices where patches were extracted.
        col_idx (torch.Tensor): Column indices where patches were extracted.
        r (int): Radius of the neighborhood.
    """
    n_patches = len(row_idx)
    fig, axes = plt.subplots(1, n_patches, figsize=(15, 5))
    tmp = torch.zeros(img.shape)  # Create a temporary mask for visualization
    for i in range(n_patches):
        tmp[row_idx[i] - r:row_idx[i] + r + 1, col_idx[i] - r:col_idx[i] + r + 1] = patches[i]
        axes[i].imshow(tmp, cmap='gray')
        axes[i].set_title(f'Patch at ({row_idx[i]}, {col_idx[i]})')
        axes[i].axis('off')
    plt.tight_layout()
    visualize_3_images(img,tmp,tmp-img, "Original Image", "Patched Image", "Patched - Original")



def creat_idx_batches_for_parl_sum(row_idx, col_idx, r, step):

    init_rows = []
    init_cols = []
    nb_batches_1d = (2*r+1)//step+(1 if (2*r+1)%step!=0 else 0) # Number of batches in one dimension
    batch_id_init_row_idx = torch.arange(nb_batches_1d) * step + min(row_idx)# Initial indices for the batches
    batch_id_init_col_idx = torch.arange(nb_batches_1d) * step + min(col_idx)  # Initial indices for the batches
    init_rows, init_cols = torch.meshgrid([batch_id_init_row_idx,batch_id_init_col_idx], indexing='ij')  # Create a grid of initial points
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
    """
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
    """

    nb_of_elements_in_batches = [len(batched_row_idxs[i]) for i in range(len(batched_row_idxs))]  # Number of elements in each batch
    nb_of_elements_in_batches = torch.sum(torch.tensor(nb_of_elements_in_batches))  # Total number of elements in all batches
    if nb_of_elements_in_batches != len(row_idx):
        raise ValueError(f"Number of elements in batches {nb_of_elements_in_batches} is not equal to number of grid points {len(row_idx)}")
    return batched_row_idxs, batched_col_idxs, bases_idxs

def check_if_patches_overlap(img,row_idxs, col_idxs, r, step):
    """ Check if patches overlap when placed at specified row and column indices."""
    img = torch.zeros(img.shape)
    row_grid, col_grid = torch.meshgrid([torch.arange(-r, r + 1), torch.arange(-r, r + 1)])
    row_grid = row_grid.unsqueeze(0).expand(len(row_idxs), -1, -1)
    col_grid = col_grid.unsqueeze(0).expand(len(row_idxs), -1, -1)
    row_grid = row_grid + row_idxs.unsqueeze(1).unsqueeze(2)
    col_grid = col_grid + col_idxs.unsqueeze(1).unsqueeze(2)
    for i in range(len(row_idxs)):
        img[row_grid[i], col_grid[i]] += 1  # Add the patch to the image
    vizualize_1_image(img, "Image with Patches Overlap Check")
    if torch.max(img) > 1:
        print("Patches overlap detected!")
        return True
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

def sum_patches_back(img,patches, row_idx, col_idx, r):
    """
    Sum the patches back to the image at specified row and column indices.

    Args:
        patches (torch.Tensor): Patches tensor of shape (N, 1, 2*r+1, 2*r+1).
        row_idx (torch.Tensor): Row indices where patches should be placed.
        col_idx (torch.Tensor): Column indices where patches should be placed.
        r (int): Radius of the neighborhood.

    Returns:
        torch.Tensor: The summed image with patches added back.
    """
    img_new = torch.zeros_like(img)  # Create a copy of the image to avoid modifying the original
    # Create a copy of the image to avoid modifying the original
    normalizer = torch.zeros_like(img)  # Create a normalizer to avoid double counting
    row_grid, col_grid = torch.meshgrid([torch.arange(-r, r + 1), torch.arange(-r, r + 1)])
    for i in range(len(row_idx)):
        img_new[row_grid+row_idx[i], col_grid+col_idx[i]] += patches[i]  # Add the patch to the image
        normalizer[row_grid+row_idx[i], col_grid+col_idx[i]] += 1  # Increment the normalizer at the patch location
    img_new[normalizer != 0]  /= normalizer[normalizer != 0] # Normalize the summed image to avoid double counting

    return img_new, normalizer

if __name__ == "__main__":
    images, _ = init_read_data(20)
    img = images[0, 0]  # Take the first image from the batch
    img = img.squeeze()  # Remove the channel dimension
    img = img/ torch.max(img)
    step = 2  # Step size for downsampling the mask
    r = 3  # Radius of the neighborhood
    mask = img>0.0  # Create a mask of the same size as the image
    row_idxs, col_idxs = extract_grid_centers_from_mask(mask, r,step)  # Extract grid centers from the mask
    visualize_sampling_grid(mask, row_idxs, col_idxs)  # Visualize the sampling grid
    st_time = time.time()  # Start the timer
    img_batch = extract_patches_from_mask(img, row_idxs, col_idxs, r)  # Extract pixels in the mask from the image
    print(f"Time to extract patches size{2*r+1}, # of grid points {len(row_idxs)}: {time.time() - st_time:.4f} seconds")
    visualize_patches(img,img_batch,row_idxs, col_idxs, r)
    img_reconstructed, normalizer = sum_patches_back(img, img_batch, row_idxs, col_idxs, r)  # Sum the patches back to the image
    visualize_3_images(img, img_reconstructed,img_reconstructed-img, "Original Image", "Reconstructed Image", "Reconstruction - Original", "Summation")
    vizualize_1_image(normalizer, "Normalizer Image for Normal Summation")
    row_idxs, col_idxs, bases_idxs = creat_idx_batches_for_parl_sum(row_idxs, col_idxs, r, step)  #
    for i in range(len(row_idxs)):
        check_if_patches_overlap(img,row_idxs[i],col_idxs[i], r, step)
    img_reconstructed, normalizer = sum_patches_back_batched(img,img_batch,row_idxs,col_idxs,bases_idxs,r)# Create batches of indices for parallel summation
    visualize_3_images(img, img_reconstructed, img_reconstructed - img, "Original Image", "Reconstructed Image",
                       "Reconstruction - Original", "Batched Summation")
    vizualize_1_image(normalizer, "Normalizer Image for Batched Summation")
    plt.show()
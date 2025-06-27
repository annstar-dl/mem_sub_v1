import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import time
from testing_with_matlab import visualize_2_images, visualize_3_images, visualize_sampling_grid

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
    mnist_test = datasets.MNIST(root='/home/astar/Projects/VesicleSegment/data', train=True, download=True, transform=transform)
    test_loader = DataLoader(mnist_test, batch_size=nb_images, shuffle=False)
    # Get one batch of 20 images and labels
    images, labels = next(iter(test_loader))
    return images, labels

def extract_grid_centers_from_mask(mask, r):
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
    tmp[::4, ::4] = 1  # Downsample the mask to reduce the number of points
    tmp[mask != 1] = 0  # Set the mask points to 1
    row_idx, col_idx = torch.where(tmp == 1)

    # Filter out indices that are too close to the edges
    valid_mask = (row_idx >= r) & (row_idx < mask.shape[0] - r) & \
                 (col_idx >= r) & (col_idx < mask.shape[1] - r)

    return row_idx[valid_mask], col_idx[valid_mask]

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
    img_new = img.detach().clone()  # Create a copy of the image to avoid modifying the original
    # Create a copy of the image to avoid modifying the original
    normalizer = torch.zeros_like(img)  # Create a normalizer to avoid double counting
    row_grid, col_grid = torch.meshgrid([torch.arange(-r, r + 1), torch.arange(-r, r + 1)])
    row_grid = row_grid.unsqueeze(0).expand(len(row_idx), -1, -1)
    col_grid = col_grid.unsqueeze(0).expand(len(row_idx), -1, -1)
    row_grid = row_grid + row_idx.unsqueeze(1).unsqueeze(2)
    col_grid = col_grid + col_idx.unsqueeze(1).unsqueeze(2)
    batch_size = 5
    batch_idx = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).expand(-1, row_grid.shape[1], row_grid.shape[2])
    tmp = torch.zeros((batch_size, img.shape[-2], img.shape[-1]))
    tmp_norm = torch.zeros((batch_size, img.shape[-2], img.shape[-1]))

    for i in range(0, len(row_idx), batch_size):
        end_idx = min(i + batch_size, len(row_idx))
        nb_elem = end_idx - i
        tmp[batch_idx[:min(batch_size,nb_elem)],row_grid[i:end_idx],col_grid[i:end_idx]] = patches[i:end_idx]
        tmp_norm[batch_idx[:min(batch_size, nb_elem)], row_grid[i:end_idx], col_grid[i:end_idx]] = 1
        img_new+=torch.sum(tmp,dim=0)
        normalizer+=torch.sum(tmp_norm,dim=0)   # Increment the normalizer for each patch added
        tmp[...] = 0  # Reset the temporary tensor for the next batch
        tmp_norm[...] = 0  # Reset the temporary normalizer for the next batch
    normalizer[normalizer == 0] = 1  # Avoid division by zero
    img_new /= normalizer # Normalize the summed image to avoid double counting

    return img_new, normalizer

if __name__ == "__main__":
    images, _ = init_read_data(20)
    img = images[0, 0]  # Take the first image from the batch
    img = img.squeeze()  # Remove the channel dimension
    img = img/ torch.max(img)

    r = 1  # Radius of the neighborhood
    mask = img>0.0  # Create a mask of the same size as the image
    row_idx, col_idx = extract_grid_centers_from_mask(mask, r)  # Extract grid centers from the mask
    visualize_sampling_grid(mask, row_idx, col_idx)  # Visualize the sampling grid
    st_time = time.time()  # Start the timer
    img_batch = extract_patches_from_mask(img, row_idx, col_idx, r)  # Extract pixels in the mask from the image
    print(f"Time to extract patches size{2*r+1}, # of grid points {len(row_idx)}: {time.time() - st_time:.4f} seconds")
    visualize_patches(img,img_batch,row_idx, col_idx, r)
    img_reconstructed, normalizer = sum_patches_back(img, img_batch, row_idx, col_idx, r)  # Sum the patches back to the image
    visualize_3_images(img, img_reconstructed,img_reconstructed-img, "Original Image", "Reconstructed Image", "Reconstruction - Original")
    vizualize_1_image(normalizer,"Normalizer Image")
    plt.show()
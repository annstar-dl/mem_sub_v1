import torch
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import kornia
import time

def visualize_mnist_images(images, labels, title='MNIST Images'):
    # Plot the images in a 4x5 grid
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

def init_read_data(nb_images):
    # Initialize the data reading process
    print("Reading MNIST dataset...")
    # Define a transform to convert images to tensors
    transform = transforms.Compose([transforms.Resize((30,30)),transforms.ToTensor()])
    # Download the MNIST test dataset
    mnist_test = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_loader = DataLoader(mnist_test, batch_size=nb_images, shuffle=False)
    # Get one batch of 20 images and labels
    images, labels = next(iter(test_loader))
    return images, labels

def rotate_images(images, labels, angle):
    # Rotate images by a specified angle
    rotated_images = []
    for img in images:
        img = transforms.functional.rotate(img, angle)
        rotated_images.append(img)
    return torch.stack(rotated_images), labels


def rotate_images_kornia(images, angle):
    """    Rotate a batch of images using Kornia.
    Args:
        images (torch.Tensor): Batch of images with shape (B, C, H, W).
        angle_degrees (float or list): Angle in degrees to rotate the images, or list of angles for each image.

        If you are using kornia in your research-related documents, it is recommended that you cite the paper.
        @inproceedings{eriba2019kornia,
          author    = {E. Riba, D. Mishkin, D. Ponsa, E. Rublee and G. Bradski},
          title     = {Kornia: an Open Source Differentiable Computer Vision Library for PyTorch},
          booktitle = {Winter Conference on Applications of Computer Vision},
          year      = {2020},
          url       = {https://arxiv.org/pdf/1910.02190.pdf}
    }
    """
    b = images.size(0) # Get the batch size
    # Resize the angle to match the batch size
    if isinstance(angle, (int, float)):
        angle = torch.tensor([angle] * b,dtype=torch.float32)  # Repeat the angle for each image in the batch
    elif isinstance(angle, list) and len(angle) != b:
        raise ValueError(f"Expected angle_degrees to be a single value or a list of length {b}, got {len(angle)}")

    # define the rotation center
    center = torch.ones(b, 2)
    center[...,0] = images.size(2) / 2
    center[...,1] = images.size(3) / 2
    # Rotate the batch
    rotated_images = kornia.geometry.rotate(images, angle, center)
    return rotated_images

if __name__ == "__main__":
    # Initialize and read the data
    nb_vesicles = 100
    nb_nodes_per_vesicle = 350
    nb_image = nb_vesicles * nb_nodes_per_vesicle # Number of images to read
    images, labels = init_read_data(nb_image)
    # Visualize the images
    visualize_mnist_images(images, labels, "Original MNIST Images")
    print("MNIST dataset visualization completed.")
    # Rotate images using kornia
    start_time = time.time()
    for angle in range(1,10):
        rotated_images = rotate_images_kornia(images, angle)
    end_time = time.time() -start_time
    print(f"Total time for rotating images: {end_time:.2f} seconds")
    # Visualize the rotated images
    visualize_mnist_images(rotated_images, labels, f"Rotated MNIST Images by {angle} degrees")
    plt.show()



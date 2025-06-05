import torch
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import kornia
import time
from image_rotation import rotate_images_kornia, init_read_data, visualize_mnist_images


def calculate_mse_loss(original, rotated):
    """Calculate the Mean Squared Error (MSE) loss between original and rotated images."""
    return torch.mean((original - rotated) ** 2, dim=(1, 2, 3))  # MSE across batch, height, and width

def find_best_rotation_angle(original_images, rotated_images, angle_range=(0, 180),angle_step=1):
    """Find the best rotation angle that minimizes the MSE loss."""
    b = original_images.size(0)  # Get the batch size

    angles_list = list(range(angle_range[0], angle_range[1],angle_step))
    loss = torch.zeros((b,len(angles_list)), dtype=torch.float32) + float('inf')

    for angle_idx, angle in enumerate(angles_list):  # Test angles from 0 to 180 degrees
        rotated = rotate_images_kornia(rotated_images, angle)
        loss[:,angle_idx] = calculate_mse_loss(original_images, rotated)

    loss_agr_min_idx = torch.argmin(loss,dim=1)

    best_angles = torch.tensor(angles_list, dtype=torch.float32, device=original_images.device)
    best_angles = best_angles[loss_agr_min_idx]  # Get the best angles for each image
    return best_angles


def best_rotation_simulation(nb_images=5,angle_range=(-180, 180),angle_step=1):
    """Simulate finding the best rotation angle for a batch of images."""
    # Initialize and read data
    initial_images, labels = init_read_data(nb_images)
    visualize_mnist_images(initial_images,labels, title='Initial MNIST Images')
    random_secret_rotation = torch.randint(angle_range[0],angle_range[1], (nb_images,), dtype=torch.float32)  # Random rotation angles between 0 and 180 degrees
    random_secret_rotation += torch.randn((nb_images,),dtype=torch.float32) * 0.1  # Add some noise to the rotation angles
    # Rotate images by 45 degrees
    rotated_images = rotate_images_kornia(initial_images, random_secret_rotation)
    visualize_mnist_images(rotated_images, labels, title='Rotated MNIST Images')
    # Find the best rotation angle
    best_angle = find_best_rotation_angle(rotated_images, initial_images, angle_range=angle_range, angle_step=angle_step)
    new_rotated_images = rotate_images_kornia(initial_images, best_angle)
    visualize_mnist_images(new_rotated_images, labels, title='Best Found Rotation MNIST Images')

    for i in range(nb_images):
        print(f"Image {i+1}: Original angle: {random_secret_rotation[i].item():.2f} degrees, Best found angle: {best_angle[i].item():.2f} degrees")

if __name__ == "__main__":
    # Run the best rotation simulation
    nb_vesicles = 1
    nb_nodes_per_vesicle = 20
    nb_images = nb_vesicles * nb_nodes_per_vesicle  # Number of images to read
    angle_range = (-180, 180)  # Define the range of angles to test
    angle_step = 1  # Step size for angle increments
    best_rotation_simulation(nb_images, angle_range,angle_step)
    print("Best rotation angle simulation completed.")
    plt.show()



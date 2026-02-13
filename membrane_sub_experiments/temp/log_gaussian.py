from PIL import Image, ImageFilter
import os

def denoise_image(image_path, output_path, gauss_radius=1.0):
    """
    Denoise an image using a simple Gaussian blur.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the denoised image.
        denoise_strength (float): Strength of the denoising effect.
    """
    # Open the image
    image = Image.open(image_path).convert("RGB")

    # Apply Gaussian blur for denoising
    denoised_image = image.filter(ImageFilter.GaussianBlur(radius=(gauss_radius,gauss_radius)))

    # Save the denoised image
    denoised_image.save(output_path)
    print(f"Denoised image saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    smoothing_radius = 1.0
    data_dir = r"/home/astar/Projects/vesicles_data"
    image_name = r"slot6_100_0002ms"
    image_path = os.path.join(data_dir,"test",image_name+".jpg")
    output_path = os.path.join(data_dir,"test_smoothed")  # Replace with your desired output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path, image_name + ".png")
    denoise_image(image_path, output_path, gauss_radius=smoothing_radius)
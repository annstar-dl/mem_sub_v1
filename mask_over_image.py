from PIL import Image
import os

def mask_to_green_rgba(mask_path, alpha = 0.3):
    # Open mask as grayscale
    alpha = int(alpha * 255)  # Convert alpha to 0-255 range
    mask = Image.open(mask_path).convert("L")
    mask = mask.point(lambda x: alpha if x > 0 else 0, 'L')  # Convert to binary mask
    # Create green image (R=0, G=255, B=0)
    green = Image.new("RGBA", mask.size, (0, 255, 0, 0))
    # Use mask as alpha channel
    green.putalpha(mask)
    return green

def mask_over_image(image_path, mask_path, output_path):
    """
    Apply a mask over an image and save the result.

    Args:
        image_path (str): Path to the input image.
        mask_path (str): Path to the mask image.
        output_path (str): Path to save the masked image.
    """
    # Open the images
    image = Image.open(image_path).convert("RGBA")
    mask = mask_to_green_rgba(mask_path)

    # Ensure the mask is the same size as the image
    mask = mask.resize(image.size)

    # Apply the mask
    overlayed = Image.alpha_composite(image,mask)

    # Save the result
    overlayed.save(output_path)
    print(f"Masked image saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    data_dir = r"/home/astar/Projects/vesicles_data"
    image_name = r"slot6_100_0002ms"
    image_path = os.path.join(data_dir,"test",image_name+".jpg")
    mask_path = os.path.join(data_dir,"labels",image_name+".png")
    output_path = os.path.join(data_dir,"test_overlayed_labels")  # Replace with your desired output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path, image_name + ".png")
    mask_over_image(image_path, mask_path, output_path)
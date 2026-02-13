from PIL import Image
import os
import numpy as np

def mask_to_color_rgba(mask_path, alpha = 0.3, color=(0,255,0)):
    # Open mask as grayscale
    alpha = int(alpha * 255)  # Convert alpha to 0-255 range
    mask = Image.open(mask_path).convert("L")
    mask = mask.point(lambda x: alpha if x > 0 else 0, 'L')  # Convert to binary mask
    # Create green image (R=0, G=255, B=0)
    green = Image.new("RGBA", mask.size, color+(0,))
    # Use mask as alpha channel
    green.putalpha(mask)
    return green

def mask_over_image(image_path, gt_path, pred_path, output_path):
    """
    Apply a mask over an image and save the result.

    Args:
        image_path (str): Path to the input image.
        mask_path (str): Path to the mask image.
        output_path (str): Path to save the masked image.
    """
    # Open the images
    image = Image.open(image_path).convert("RGBA")
    gt = mask_to_color_rgba(gt_path, color=(255,0,0))
    pred = mask_to_color_rgba(pred_path, color=(0,255,0))

    # Apply the mask
    overlayed = Image.alpha_composite(image,gt)
    overlayed = Image.alpha_composite(overlayed,pred)

    # Save the result
    overlayed.save(output_path)
    print(f"Masked image saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    img_dir = r"/home/astar/Projects/membrane_detection/data/membrane/images/test"
    gt_dir =  r"/home/astar/Projects/membrane_detection/data/membrane/labels/test"
    pred_dir =  r"/home/astar/Projects/vesicles_data/iclr_experiments/test/labels"
    output_dir = r"/home/astar/Projects/vesicles_data/iclr_experiments/test/overlayed_labels"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for fname in os.listdir(gt_dir):
        basename, ext = os.path.splitext(fname)
        img_path = os.path.join(img_dir,basename+".jpg")
        gt_path = os.path.join(gt_dir,fname)
        pred_path = os.path.join(pred_dir,fname)

        output_path = os.path.join(output_dir, fname)
        mask_over_image(img_path, gt_path, pred_path, output_path)
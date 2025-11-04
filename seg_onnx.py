import os.path
import onnxruntime
import numpy as np
from scipy.ndimage import gaussian_filter
import argparse
from PIL import Image
import os

# Check if a GPU is available
if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
    print("Using CUDAExecutionProvider")
    providers = ['CUDAExecutionProvider']
else:
    print("CUDAExecutionProvider not available, falling back to CPU")
    providers = ['CPUExecutionProvider']

def standartize(img, sigma=24.):
    # Apply Gaussian blur
    if img.ndim == 3:
        smooth = np.stack([gaussian_filter(img[..., c], sigma) for c in range(img.shape[2])], axis=2)
    else:
        smooth = gaussian_filter(img, sigma)
    img = np.subtract(img, smooth)
    del smooth
    # scale pixel intensities
    img /= np.std(img)
    return img

def load_img_paths(data_path):
    """Load image paths from a directory."""

    img_paths = []
    print(f"Loading images from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    for root, _, files in os.walk(data_path):
        print(f"Searching for images in: {root}")
        # Filter files to include only image files
        print(f"Files found: {files}")
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_paths.append(os.path.join(root, file))
    return img_paths

def save_output_as_label(output, fpath):
    """Save the output array as an image."""
    output = np.squeeze(output)
    output = output.astype(np.uint8)
    img = Image.fromarray(output, mode='L')
    img.save(fpath)

def save_output_as_image(output, fpath):
    """Save the output array as an image."""
    output = np.squeeze(output)
    output = (output - output.min()) / (output.max() - output.min()) * 255.0  # Normalize to [0, 255]
    output = output.astype(np.uint8)
    img = Image.fromarray(output, mode='L')
    img.save(fpath)

def process_dir(args):
    """Process all image files in a folder."""
    sess = onnxruntime.InferenceSession(args.onnx_model_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    fpaths = load_img_paths(args.data_path)
    print(fpaths)
    for fpath in fpaths:
        img = np.array(Image.open(fpath), dtype=np.float32)
        img = standartize(img)
        img = np.stack([img, img, img], axis=0) if img.ndim == 2 else img
        img = np.expand_dims(img, axis=0)
        output = sess.run(None, {input_name: img})
        filename = os.path.basename(fpath)
        filename = os.path.splitext(filename)[0] + '.png'
        print(f"Processing {filename}...")
        save_output_as_image(output, os.path.join(args.output_dir_image, filename))
        save_output_as_label(output,os.path.join(args.output_dir_label, filename))

def process_file(args):
    """Process a single image file."""
    sess = onnxruntime.InferenceSession(args.onnx_model_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    fpath = os.path.join(args.data_path, args.file_name)
    img = np.array(Image.open(fpath), dtype=np.float32)
    img = standartize(img)
    img = np.stack([img, img, img], axis=0) if img.ndim == 2 else img
    img = np.expand_dims(img, axis=0)
    output = sess.run(None, {input_name: img})
    filename = os.path.basename(fpath)
    filename = os.path.splitext(filename)[0] + '.png'
    print(f"Processing {filename}...")
    save_output_as_image(output, os.path.join(args.output_dir_image, filename))
    save_output_as_label(output,os.path.join(args.output_dir_label, filename))

        
if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Run ONNX model inference on images")
    args.add_argument("--model_dir", type=str, help="Directory where the ONNX model is located")
    args.add_argument('--onnx_fname', type=str, required=True, help='Path to the ONNX model file')
    args.add_argument('--data_path', type=str, required=True, help='Path to the directory containing images')
    args.add_argument("--save_dir", type=str, default=None, help="Directory to save the output images")
    args.add_argument("-fn","--file_name", type=str, default=None, help="Name of file to convert (default: None), if None process all files in the folder")
    args = args.parse_args()
    args.onnx_model_path = os.path.join(args.model_dir, args.onnx_fname)
    args.output_dir_label = os.path.join(args.save_dir, 'labels')
    args.output_dir_image = os.path.join(args.save_dir, 'labels_pseudcolor')
    os.makedirs(args.output_dir_label, exist_ok=True)
    os.makedirs(args.output_dir_image, exist_ok=True)
    if args.file_name is None:
        process_dir(args)
    else:
        process_file(args)

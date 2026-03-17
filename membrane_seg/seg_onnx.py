import os.path
import onnxruntime
import numpy as np
from scipy.ndimage import gaussian_filter
import argparse
from PIL import Image
import os
import time
import cv2
# Check if a GPU is available
if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
    print("Using CUDAExecutionProvider")
    providers = ['CUDAExecutionProvider']
else:
    print("CUDAExecutionProvider not available, falling back to CPU")
    providers = ['CPUExecutionProvider']


class Standardize:
    """
    Standardizes an image by zero centering and scales pixel intensities

    Args:
        sigma(float, optional): The Gaussian sigma value to be used. Default: 24.

    Raises:
        ValueError: When sigma is not a float.
    """

    def __init__(self, sigma=24.):
        if not isinstance(sigma, float):
            raise ValueError("sigma is invalid. It should be a float")
        self.sigma = sigma

    def __call__(self, data):

        # zero center pixels
        smooth = cv2.GaussianBlur(data['img'], (0, 0), sigmaX=self.sigma)
        # ksize=(0,0) - kernel size derived from sigmaX
        # sigmaY=None - set to sigmaX
        # default borderType equal to cv2.BORDER_REFLECT_101
        # src - https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
        data['img'] = np.subtract(data['img'], smooth)
        del smooth
        # scale pixel intensities
        data['img'] /= np.std(data['img'])

        return data

class StandardizeMAD:
    """
    Standardizes an image by zero centering and scales pixel intensities using Median Absolute Deviation (MAD)

    Args:
        None

    """
    def __init__(self, sigma=30.):
        if not isinstance(sigma, float):
            raise ValueError("sigma is invalid. It should be a float")
        self.sigma = sigma

    def __call__(self, data):
        # zero center pixels
        smooth = cv2.GaussianBlur(data['img'], (0, 0), sigmaX=self.sigma)
        data['img'] = np.subtract(data['img'], smooth)
        # scale pixel intensities using MAD
        mad = np.median(np.abs(data['img'] - np.median(data['img'])))
        data['img'] = data['img'] / (mad + 1e-6)
        return data

def standardize(img, standardize_type):
    data = {'img': img}
    if standardize_type == 'Standardize':
        preprocess_op = Standardize()
    elif standardize_type == 'StandardizeMAD':
        preprocess_op = StandardizeMAD()
    else:
        raise ValueError(f"Unknown standardization type: {standardize_type}")
    img = preprocess_op(data)['img']
    return img

def load_img_paths(data_path):
    """Load image paths from a directory."""
    # Check if data_path is a file
    if os.path.isfile(data_path):
        if data_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            return [data_path]
        else:
            raise ValueError(f"File {data_path} is not a supported image format.")
    else:
        img_paths = []
        print(f"Loading images from: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        for root, _, files in os.walk(data_path):
            print(f"Searching for images in: {root}")
            # Filter files to include only image files
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

def read_config(config_path):
    """Read configuration from a YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_standardization_type(config):
    """
    Descend `data` using `keys` (list of str or int).
    If current level is a list and the next key is a str, it searches
    the first dict in the list that contains that key.
    """
    config = config.get('Deploy', {})
    config = config.get('transforms', {})[0]
    return config.get('type', None)

def process(args):
    """Segment membranes of image files in a folder or a single file."""
    sess_options = onnxruntime.SessionOptions()
    # Set this to the number of CPUs requested in your Slurm job (e.g., via --cpus-per-task)
    # If you aren't sure, setting it to 4 or 8 is usually safe.
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 4
    # Create ONNX Runtime session
    sess = onnxruntime.InferenceSession(args.onnx_model_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    fpaths = load_img_paths(args.data_path)
    #read standartization type from config file
    config = read_config(os.path.join(args.model_dir, 'deploy.yaml'))
    standard_type = get_standardization_type(config)

    for fpath in fpaths:
        start_time = time.time()
        img = np.array(Image.open(fpath), dtype=np.float32)
        img = standardize(img, standard_type)
        img = np.stack([img, img, img], axis=0) if img.ndim == 2 else img
        img = np.expand_dims(img, axis=0)
        output = sess.run(None, {input_name: img})
        filename = os.path.basename(fpath)
        filename = os.path.splitext(filename)[0] + '.png'
        save_output_as_image(output, os.path.join(args.output_dir_label, filename))
        #save_output_as_label(output,os.path.join(args.output_dir_label, filename))
        print(f"Processing time of {fpath}: {time.time() - start_time}")



        
if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Run ONNX model inference on images")
    args.add_argument("--model_dir", type=str, help="Directory where the ONNX model is located")
    args.add_argument('--onnx_fname', type=str, required=True, help='Path to the ONNX model file')
    args.add_argument('--data_path', type=str, required=True, help='Path to the directory containing images or image file')
    args.add_argument("--save_dir", type=str, default=None, help="Directory to save the output images")
    args = args.parse_args()
    args.onnx_model_path = os.path.join(args.model_dir, args.onnx_fname)
    args.output_dir_label = os.path.join(args.save_dir, 'labels')
    os.makedirs(args.output_dir_label, exist_ok=True)
    process(args)


import os, yaml
import numpy as np
from PIL import Image


def read_parameters_from_yaml_file():
    """
    Read a YAML configuration file and return its contents as a dictionary.

    Returns:
        dict: Contents of the YAML file.
    """
    filename = '../../../parameters.yml'  # Replace with your YAML file path
    maindir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
    filepath = os.path.join(maindir, filename)  # Construct the full path to the YAML file
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config


def read_img(fpath, mask=False):
    #check if the file exists
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"File {fpath} does not exist.")
    else:
        img = Image.open(fpath)
        img = np.array(img,dtype = np.float64)
        if mask:
            img = img - np.min(img)
            if np.max(img)==0:
                raise ValueError(f"Empty mask file, max value is zero in {fpath}")
            img = img / np.max(img)
            img = ( img > 0.5 ).astype(np.float64)
    return img


def save_im(img, fpath):
    """Save the image to a file after normalizing it to the range [0, 255].
    Args:
        img (numpy.ndarray): Image array to save.
        fpath (str): Path to save the image file.
    """
    img = img - np.min(img)
    img = img / np.max(img) * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img,"L")
    img.save(fpath)

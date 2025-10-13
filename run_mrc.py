import os
import numpy as np
from PIL import Image
from membrane_subtract_mrc import membrane_subtract
from tqdm import tqdm
import argparse
from scipy.io import savemat
from mrc_utils import load_mrc, downsample_micrograph, save_im_mrc_same_size, \
    upsample_micrograph, new_ds_shape
from utils import read_dict_from_yaml_file

def read_img(fpath, mask=False):
    #check if the file exists
    print(f"Processing image: {fpath}")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"File {fpath} does not exist.")
    else:
        img = Image.open(fpath)
        img = np.array(img,dtype = np.float64)
        if mask:
            img = img - np.min(img)
            img = img / np.max(img)
            img = (img >0.5).astype(np.float64)
    return img

def read_mrc(fpath):
    #check if the file exists
    print(f"Processing image: {fpath}")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"File {fpath} does not exist.")
    img, header, voxel_size = load_mrc(fpath)
    img = img.astype(np.float64)
    return img, header, voxel_size

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

def main(args):
    imgs_path = os.path.join(args.dataset_path, args.imgs_dir)
    masks_path = os.path.join(args.dataset_path, args.masks_dir)
    for fpath in tqdm(os.listdir(imgs_path), desc="Processing images"):
        if not fpath.endswith("mrc"):
            continue
        basename = os.path.splitext(fpath)[0]
        #scip if the output files already exist
        if os.path.exists(os.path.join(args.imgsout_path + "_jpeg", basename + ".jpeg")) and \
              os.path.exists(os.path.join(args.imgsout_path + "_mat", basename + ".mat")) and \
                os.path.exists(os.path.join(args.imgsout_path + "_mrc", basename + ".mrc")):
            print(f"Output for {basename} already exists. Skipping...")
            continue

        img_fname = basename+".mrc"
        # read radius size from parameters file
        parameters = read_dict_from_yaml_file()
        border = parameters["r"]  # Radius of neighboring around grid point
        img, header, voxel_size = read_mrc(os.path.join(imgs_path, img_fname))
        mask = read_img(os.path.join(masks_path, basename + ".png"),True)
        padded_org_shape, ds_shape, ds_factor = new_ds_shape(img.shape, voxel_size[0])
        img_downsampled = downsample_micrograph(img, padded_org_shape,ds_shape,ds_factor,border, "center")
        # check if the image is the same size as the mask
        if img_downsampled.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image {img_fname} and mask {basename}.png must have the same dimensions. "
                f"Image shape: {img.shape}, Mask shape: {mask.shape}")
        # run membrane subtraction algorithm
        imgout_ds = membrane_subtract(img_downsampled, mask)
        #upsample the membrane estimate to the original size
        imgout = upsample_micrograph(imgout_ds, img.shape, padded_org_shape, border, "center")
        sub_img = img - imgout
        # add background back to the subtracted image
        if "mat" in args.out_format:
            # Save as .mat file if specified
            savemat(os.path.join(args.imgsout_path+"_mat", basename + ".mat"),
                    {'img': img, 'label': mask, 'mem': imgout, 'sub': sub_img})
        if "jpeg" in args.out_format:
            # Save as .png file if specified
            save_im(sub_img, os.path.join(args.imgsout_path+"_jpeg", basename + ".jpeg"))
        if "png" in args.out_format:
            # Save as .png file if specified
            save_im(sub_img, os.path.join(args.imgsout_path+"_png", basename + ".png"))
        if "mrc" in args.out_format:
            # Save as .png file if specified
            save_im_mrc_same_size(sub_img, os.path.join(args.imgsout_path+"_mrc", basename + ".mrc"), header)
        if args.save_reconstruction:
            save_im(imgout, os.path.join(args.imgsout_reconstructed_path, basename + ".tif"))
            np.save(os.path.join(args.imgsout_reconstructed_path, basename + ".npy"), imgout)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Process membrane images and subtract membranes.")
    parser.add_argument("-dp","--dataset_path", type=str,help="Directory containing folders with images and labels")
    parser.add_argument("-id","--imgs_dir", type=str, default="images", help="Directory containing images.")
    parser.add_argument("-md","--masks_dir", type=str, default="labels", help="Directory containing labels for images.")
    parser.add_argument("-s","--sigma", type=float, default=24.0, help="Sigma for Gaussian filter to flatten background.")
    parser.add_argument("--in_format", type=str,choices=["mrc","jpeg","jpg","tif","tiff"],help="Format of input images. Default is mrc", default="mrc")
    parser.add_argument("--out_format", nargs="+", default="png", help="List of file format to save subracted images. Choices are .mat,.png.,mrc formats")
    parser.add_argument("-ds","--downsample_factor", type=int, default=1, help="Factor by which to downsample the images (default: 1).")
    parser.add_argument("--save_reconstruction", action="store_true", default=False, help="Save the reconstructed membranes.")
    args = parser.parse_args()

    main_path = args.dataset_path
    imgout_mainpath = os.path.join(main_path, "reconstructions")
    imgsout_subracted_path = os.path.join(imgout_mainpath,"subtracted")


    if args.save_reconstruction:
        imgsout_reconstructed_path = os.path.join(imgout_mainpath, "reconstructed_membranes")
        if not os.path.exists(imgsout_reconstructed_path):
            os.makedirs(imgsout_reconstructed_path)
        args.imgsout_reconstructed_path = imgsout_reconstructed_path
    args.imgsout_path = imgsout_subracted_path
    for fmt in args.out_format:
        imgsout_path = imgsout_subracted_path + f"_{fmt}"
        if not os.path.exists(imgsout_path):
            os.makedirs(imgsout_path)
    main(args)




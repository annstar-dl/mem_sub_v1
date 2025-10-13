import os
import numpy as np
from PIL import Image
from temp.membrane_subtract import membrane_subtract
from tqdm import tqdm
import argparse
from scipy.io import savemat
from skimage import transform



def read_img(fpath,in_format,downsample_factor=1, mask=False):
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
        if downsample_factor > 1:
            img = transform.rescale(img, 1/downsample_factor, anti_aliasing=True, preserve_range=True)
    return img


def save_im(img, fpath):
    """Save the image to a file after normalizing it to the range [0, 255].
    Args:
        img (numpy.ndarray): Image array to save.
        fpath (str): Path to save the image file.
    """
    img = img.detach().cpu().numpy()
    img = img.astype(np.uint8)
    img = Image.fromarray(img,"L")
    img.save(fpath)

def main(args):
    imgs_path = os.path.join(args.dataset_path, args.imgs_dir)
    masks_path = os.path.join(args.dataset_path, args.masks_dir)
    for fpath in tqdm(os.listdir(imgs_path), desc="Processing images"):
        if not fpath.endswith(args.in_format):
            continue

        basename = os.path.splitext(fpath)[0]
        if os.path.exists(os.path.join(args.imgsout_path + "_jpeg", basename + ".jpeg")) and \
              os.path.exists(os.path.join(args.imgsout_path + "_mat", basename + ".mat")):
            print(f"Output for {basename} already exists. Skipping...")
            continue

        img_fname = basename+"."+args.in_format
        img = read_img(os.path.join(imgs_path, img_fname),args.in_format, args.downsample_factor)
        mask = read_img(os.path.join(masks_path, basename + ".png"),"png",args.downsample_factor,True)[0]
        # check if the image is the same size as the mask
        if img.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image {img_fname} and mask {basename}.png must have the same dimensions. Image shape: {img.shape}, Mask shape: {mask.shape}")
        # run membrane subtraction algorithm
        imgout, sub_img = membrane_subtract(img, mask)

        # add background back to the subtracted image
        if "mat" in args.out_format:
            # Save as .mat file if specified
            savemat(os.path.join(args.imgsout_path+"_mat", basename + ".mat"),
                    {'img': img, 'label': mask, 'mem': imgout.numpy(), 'sub': sub_img.numpy()})
        if "jpeg" in args.out_format:
            # Save as .png file if specified
            save_im(sub_img, os.path.join(args.imgsout_path+"_jpeg", basename + ".jpeg"))
        if "png" in args.out_format:
            # Save as .png file if specified
            save_im(sub_img, os.path.join(args.imgsout_path+"_png", basename + ".png"))
        if args.save_reconstruction:
            save_im(imgout, os.path.join(args.imgsout_reconstructed_path, basename + ".tif"))
            np.save(os.path.join(args.imgsout_reconstructed_path, basename + ".npy"), imgout)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Process membrane images and subtract membranes.")
    parser.add_argument("-dp","--dataset_path", type=str,help="Directory containing folders with images and labels")
    parser.add_argument("-id","--imgs_dir", type=str, default="images", help="Directory containing images.")
    parser.add_argument("-md","--masks_dir", type=str, default="labels", help="Directory containing labels for images.")
    parser.add_argument("-s","--sigma", type=float, default=24.0, help="Sigma for Gaussian filter to flatten background.")
    parser.add_argument("--in_format", type=str,choices=["jpeg","jpg","tif","tiff"],help="Format of input images. Default is mrc", default="mrc")
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




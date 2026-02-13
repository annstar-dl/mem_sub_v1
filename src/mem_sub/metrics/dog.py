import numpy as np
from scipy import ndimage


def get_filter(size : int =15)->np.ndarray:
# Create a difference of gaussian filter of size
# The filter is normalized to have zero avg value
# and unit norm

    sigma1=size/3.0
    sigma2=2.0*sigma1
    # Create a grid of coordinates
    X, Y = np.meshgrid(np.linspace(-(size-1)/2, (size-1)/2, size),
                       np.linspace(-(size-1)/2, (size-1)/2, size))
    # Calculate the radius at each point
    r = np.sqrt(X**2 + Y**2)
    # Calculate f1
    f1 = np.exp(-r**2 / (2 * sigma1**2))
    f1 = f1 / np.sum(f1)
    # Calculate f2
    f2 = np.exp(-r**2 / (2 * sigma2**2))
    f2 = f2 / np.sum(f2)
    # Calculate the filter
    f = f1 - f2
    f = f / np.sqrt(np.sum(f**2))
    return f


# def dog_norm(img : np.ndarray,size: int=15)->np.float64:
# # Calculate the norm of img after filtering
#     filter_kernel = get_filter(size)
#     out = ndimage.convolve(img, filter_kernel, mode='reflect')
#     return np.sqrt(np.mean(out.flatten()**2.0))

def subtraction_metric(img : np.ndarray, sub : np.ndarray, mask, size : int=15)->np.float64:
#Calculates the efficacy of membrane subtraction
#img: original micrograph, sub: membrane subtracted image
#size: nominally set to 15, but can be increased
# Returns the fractional difference. Bigger-> more membrane subtraction
    filter_kernel = get_filter(size)
    out_img = ndimage.convolve(img, filter_kernel, mode='reflect')[mask>0]
    img_norm= np.sqrt(np.mean(out_img.flatten()**2.0))
    out_sub = ndimage.convolve(sub, filter_kernel, mode='reflect')[mask>0]
    sub_norm= np.sqrt(np.mean(out_sub.flatten()**2.0))
    metric = sub_norm/img_norm # (img_norm - sub_norm)/img_norm
    return metric, out_img, out_sub







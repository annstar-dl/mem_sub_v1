import numpy as np
from skimage.feature import hessian_matrix
from PIL import Image
import matplotlib.pyplot as plt

EPS=1e-8

def print_minmax(x):
    print(np.max(x.flatten()), np.min(x.flatten()))

def rms(x):
    return np.sqrt(np.mean(np.square(x.flatten())))

def dir_2nd_derivative(gx,gy,f_xx,f_xy,f_yy):
    trans=  f_xx * gx ** 2 + 2*f_xy * gx * gy + f_yy * gy ** 2
    perp = (-f_xx + f_yy) * gx * gy + f_xy * (gx ** 2 - gy ** 2)
    return rms(trans),rms(perp)

def get_membrane_score(img):
    gx, gy = np.gradient(img)
    grad_norm = np.sqrt(gy ** 2 + gx ** 2)
    gx, gy = gx / (grad_norm + EPS), gy / (grad_norm + EPS)
    f_xx, f_xy, f_yy = hessian_matrix(img, sigma=1)
    trans, perp = dir_2nd_derivative(gx, gy, f_xx, f_xy, f_yy)
    score = 1 - perp/(np.sqrt(trans**2 + perp**2)+EPS)
    return score


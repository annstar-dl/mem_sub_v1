import numpy as np
from scipy import special

def fuzzymask(n, r, origin=None, risetime=None):
    """
    Function that creates disk of radius r with fuzzy edges.
    Function is adapted from MATLAB code by F. Singword
    """
    if np.isscalar(n):
        n = np.array((n, n))
    elif len(n) == 1:
        n = np.array((n[0], n[0]))
    elif isinstance(n, (list, tuple)):
        n = np.array(n)

    if risetime is None:
        risetime = r / 10  # default risetime is 1/10 of the radius

    if not np.isscalar(risetime):
        raise ValueError("Risetime must be a scalar, not a tuple or an array.")

    if risetime == 0:
        k=0
    else:
        k = 1.782 / risetime

    if origin is None:
        origin = np.floor(n/2).astype(int)

    x, y = np.ogrid[-origin[0]:n[0]-origin[0], -origin[1]:n[1]-origin[1]]

    if np.isscalar(r):
        rp = np.sqrt(x**2 + y**2)
        r = (r,)
    else:
        rp = np.sqrt(x**2 + (y*r[0]/r[1])**2)

    if k==0:
        mask = (rp <= r[0]).astype(float)
    else:
        mask = 0.5 * (1 - special.erf((rp-r[0]) * k))

    return mask

if __name__ == "__main__":
    import scipy.io as sio
    from matplotlib import pyplot as plt
    n = np.array([60, 120])
    matfpath = r"/home/astar/Projects/Freds_downsampling/disk1.mat"
    # Load the .mat file into a dictionary
    mat_data = sio.loadmat(matfpath)
    n = mat_data['n']
    n = np.squeeze(n)
    mask_matlab = mat_data['msk']
    r = 0.45 * n
    risetime = 0.05*n[0]
    mask = fuzzymask(n, r, risetime=risetime)
    plt.subplot(1,3,1)
    plt.imshow(mask_matlab, cmap='gray')
    plt.title('Matlab Mask')
    plt.subplot(1,3,2)
    plt.imshow(mask, cmap='gray')
    plt.title('Python Mask')
    plt.subplot(1,3,3)
    plt.imshow(mask-mask_matlab, cmap='gray')
    plt.colorbar()
    plt.title(f'Fuzzy Mask: radius={r}, risetime={risetime}')
    plt.show()
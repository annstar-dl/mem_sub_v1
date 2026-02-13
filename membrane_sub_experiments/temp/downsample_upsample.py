from membrane_sub.mrc_tools.mrc_utils import downsample_mrc, upsample_mrc_to_original
from matplotlib import pyplot as plt
import numpy as np
import math

def create_sin_image(shape, bandwidth):
    """Create a 2D sinusoidal image."""
    if type(shape) is tuple:
        shape = list(shape)
    shape[1] = math.ceil(shape[1]/bandwidth)*bandwidth
    t = np.arange(shape[1])
    x = np.sin(2 *math.pi* t / bandwidth).astype(np.float64)
    plt.figure()
    plt.plot(x,"*-")
    x_fft = np.fft.fftshift(np.fft.fft(x))
    plt.figure()
    plt.plot(np.log(np.abs(x_fft)+1e-12), "*-")
    plt.title("FFT of 1D sinusoid")

    img = np.tile(x, (shape[0], 1))  # Repeat the row to create a 2D image
    return img.astype(np.float32)

if __name__ == "__main__":
    fpath = r"/home/astar/Projects/vesicles_data/from_Fred/subtracted/20211122/20211122/slot6_100_0001.mrc"

    #data, _, voxel_size = load_mrc(fpath)
    voxel_size = (0.1,0.1)
    data = create_sin_image((128, 128), 32) # 128, 16

    fft_line = np.fft.fftshift(np.fft.fft(data[5, :]))
    fft_freqs = np.fft.fftshift(np.fft.fftfreq(data.shape[1], 1/data.shape[1]))
    plt.figure()
    plt.plot(fft_freqs,np.abs(fft_line), "*")
    plt.title('FFT of Original Image')
    data_ds = downsample_mrc(data, voxel_size, downsample_factor=4) #4

    print("Shape ds: {}".format(data_ds.shape))
    fft_line = np.fft.fftshift(np.fft.fft(data_ds[5,:]))
    plt.figure()
    plt.plot(np.abs(fft_line),"*")
    plt.title('FFT of Original Image')
    plt.figure()
    plt.imshow(data_ds, cmap='gray')
    plt.title('Downsampled Image')
    plt.colorbar()


    print("Downsampled shape: {}".format(data_ds.shape))
    data_us = upsample_mrc_to_original(data_ds, data.shape, voxel_size, downsample_factor=2)
    print("Upsampled shape: {}".format(data_us.shape))
    plt.figure(figsize=(12, 6))
    plt.subplot(1,  2, 1)
    plt.imshow(data, cmap='gray', vmin=np.percentile(data, 1), vmax=np.percentile(data, 99))
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(data_us, cmap='gray', vmin=np.percentile(data, 1), vmax=np.percentile(data, 99))
    plt.title('Upsampled Image')
    plt.figure()
    plt.imshow(data - data_us, cmap='gray')
    plt.title('Difference Image')
    plt.colorbar()

    plt.show()


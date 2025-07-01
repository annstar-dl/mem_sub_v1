from matplotlib import pyplot as plt

def visualize_im(im, title="Image", add_figure=True):
    """
    Visualize the image tensor.
    Args:
        im (torch.Tensor): Image tensor of shape (C, H, W).
    """
    if add_figure:
        plt.figure()
    if im.dim()>2:
        im = im.permute(1, 2, 0)  # Change shape to (H, W, C) for visualization
        plt.imshow(im)
    else:
        plt.imshow(im, cmap='gray')  # For single channel images
    plt.axis('off')
    plt.title(title)

def visualize_2_images(im1, im2, title1="Image 1", title2="Image 2"):
    """
    Visualize two images side by side.
    Args:
        im1 (torch.Tensor): First image tensor of shape (C, H, W).
        im2 (torch.Tensor): Second image tensor of shape (C, H, W).
        title1 (str): Title for the first image.
        title2 (str): Title for the second image.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    visualize_im(im1, title=title1, add_figure=False)
    plt.subplot(1, 2, 2)
    visualize_im(im2, title=title2, add_figure=False)


def visualize_3_images(im1, im2, im3, title1="Image 1", title2="Image 2", title3="Image 3", suptitle="Three Images"):
    """
    Visualize three images side by side.
    Args:
        im1 (torch.Tensor): First image tensor of shape (C, H, W).
        im2 (torch.Tensor): Second image tensor of shape (C, H, W).
        im3 (torch.Tensor): Third image tensor of shape (C, H, W).
        title1 (str): Title for the first image.
        title2 (str): Title for the second image.
        title3 (str): Title for the third image.
    """
    plt.figure(figsize=(15, 5))
    plt.title(suptitle)
    plt.subplot(1, 3, 1)
    visualize_im(im1, title=title1, add_figure=False)
    plt.subplot(1, 3, 2)
    visualize_im(im2, title=title2, add_figure=False)
    plt.subplot(1, 3, 3)
    visualize_im(im3, title=title3, add_figure=False)
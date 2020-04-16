import numpy as np
import cv2
from albumentations import Resize
import matplotlib.pyplot as plt

from utils import reading


def visualize_img_mask_pair(image_3d, mask_3d, exist_label=False, height=600, width=600,
                            matplot=True, max_plot_nr=8):
    """
    Args:
    image_3d - ndarray: HxWxC image_3d
    mask_3d - ndarray: HxWxC label
    exist_label - bool: true will show labels regardless if there's any, false will only show
    images where the heart is visible
    height, width - int: displayed height and width
    matplot - bool: use matplotlib to plot (better colors imo)
    max_plot_nr - int: maximum number in matplot

    Return:
    """
    resize = Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC)
    imgs_to_plot, titles = [], []
    print("image_3d shape: ", image_3d.shape)
    print("Unique elements in mask_3d: ", np.unique(mask_3d), "\n")
    for i in range(image_3d.shape[2]):
        if exist_label or (mask_3d[:, :, i].max() > 0):

            print("Max value of slice: ", image_3d[:, :, i].max())
            print("Unique elements in slice mask: ", np.unique(mask_3d[:, :, i]))

            current_img = image_3d[:, :, i]
            current_mask = mask_3d[:, :, i]

            augmented = resize(image=current_img, mask=current_mask)
            current_img = augmented['image']
            current_mask = augmented['mask']

            current_img = (current_img * (255/current_img.max())).astype(np.uint8)
            current_mask = (current_mask * (255/mask_3d.max())).astype(np.uint8)

            if matplot:
                imgs_to_plot.extend([current_img, current_mask])
                titles.extend(["image" + str(i), "mask" + str(i)])
                if len(imgs_to_plot) >= max_plot_nr:
                    show_images(imgs_to_plot, 4, titles)
                    # input("Press any key to continue...")
                    plt.close('all')
                    imgs_to_plot, titles = [], []
            else:
                cv2.imshow("img", current_img)
                cv2.imshow("mask", current_mask)
                cv2.waitKey(0)

    cv2.destroyAllWindows()


def visualize_dataset(images_3d_path, masks_3d_path):
    """
    Args:
    images_3d_path - pathlib.Path: path to image_3ds folder
    masks_3d_path - pathlib.Path: path to mask_3d folder

    Return:
    """
    for image_3d_path in images_3d_path.glob('**/*'):
        image_3d, mask_3d = reading.get_img_mask_pair(image_3d_path, masks_3d_path)
        visualize_img_mask_pair(image_3d, mask_3d)


def show_images(images, cols=1, titles=None):
    """
    taken from https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as images.
    """
    assert((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(np.ceil(n_images/float(cols)), cols, n + 1)
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.tight_layout(pad=20)
    plt.show()

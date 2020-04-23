import numpy as np
import cv2
from albumentations import Resize
import matplotlib.pyplot as plt

import general_config


height, width, matplot = general_config.height, general_config.width, general_config.matplot
max_plot_nr, exist_label = general_config.max_plot_nr, general_config.exist_label


def visualize_img_mask_pair_2d(image, mask):
    """
    Args:
    image - ndarray: HxW image
    mask - ndarray: HxW label

    Return:
    """
    print("In visualization, original shape ", image.shape)
    resize = Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC)
    augmented = resize(image=image, mask=mask)
    image = augmented['image']
    mask = augmented['mask']

    if matplot:
        show_images([image, mask], 4, ['image', 'mask'])
        plt.close('all')
    else:
        image = (image * (255/image.max())).astype(np.uint8)
        mask = (mask * (255/mask.max())).astype(np.uint8)
        cv2.imshow("img", image)
        cv2.imshow("mask", mask)
        cv2.waitKey(0)


def visualize_img_mask_pair(image_3d, mask_3d):
    """
    Args:
    image_3d - ndarray: HxWxC image_3d
    mask_3d - ndarray: HxWxC label_3d
    Return:
    """
    resize = Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC)
    imgs_to_plot, titles = [], []
    print("In visualization, image_3d shape: ", image_3d.shape)
    print("In visualization, Unique elements in mask_3d: ", np.unique(mask_3d), "\n")
    for i in range(image_3d.shape[2]):
        if exist_label or (mask_3d[:, :, i].max() > 0):

            print("Max value of slice: ", image_3d[:, :, i].max())
            print("Unique elements in slice mask: ", np.unique(mask_3d[:, :, i]))

            current_img = image_3d[:, :, i]
            current_mask = mask_3d[:, :, i]

            augmented = resize(image=current_img, mask=current_mask)
            current_img = augmented['image']
            current_mask = augmented['mask']

            if matplot:
                imgs_to_plot.extend([current_img, current_mask])
                titles.extend(["image" + str(i), "mask" + str(i)])
                if len(imgs_to_plot) >= max_plot_nr:
                    show_images(imgs_to_plot, 4, titles)
                    # input("Press any key to continue...")
                    plt.close('all')
                    imgs_to_plot, titles = [], []
            else:
                current_img = (current_img * (255/current_img.max())).astype(np.uint8)
                current_mask = (current_mask * (255/mask_3d.max())).astype(np.uint8)
                cv2.imshow("img", current_img)
                cv2.imshow("mask", current_mask)
                cv2.waitKey(0)

    cv2.destroyAllWindows()


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

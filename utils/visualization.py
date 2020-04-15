import numpy as np
import cv2
from albumentations import Resize

from utils import reading


def visualize_img_mask_pair(image_3d, mask_3d, exist_label=False, height=600, width=600):
    """
    Args:
    image_3d - ndarray: HxWxC image_3d
    mask_3d - ndarray: HxWxC label

    Return:
    """
    resize = Resize(height=height, width=width, interpolation=cv2.INTER_CUBIC)
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

            cv2.imshow("img", current_img)
            cv2.imshow("mask", current_mask)

            # current_img_ = (image_3d[:, :, i] * (255/image_3d[:, :, i].max())).astype(np.uint8)
            # current_mask_ = (mask_3d[:, :, i] * (255/mask_3d[:, :, i].max())).astype(np.uint8)
            #
            # cv2.imshow("img_orig", current_img_)
            # cv2.imshow("mask_orig", current_mask_)
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

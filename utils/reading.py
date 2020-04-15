import numpy as np
import nibabel as nib


def get_img_mask_pair(image_path, masks_path):
    """
    Args:
    image_path - pathlib.Path: path to image
    masks_path - pathlib.Path: path to mask folder

    Finds the corresponding ground truth label

    return:
    ndarray: image and mask pair
    """
    stem, suffix = image_path.stem, image_path.suffix
    mask_path = masks_path / (stem + suffix)
    print(image_path)
    print(mask_path, "\n")

    image = np.array(nib.load(image_path).dataobj)
    mask = np.array(nib.load(mask_path).dataobj)

    return image, mask

import numpy as np
import math

import constants
"""
- % of images with existing label (not just background) (have to restructure dset)
- mean and std of images - can be done rn (by the looks i must be very careful with normalization)
- should also check out christian payer augmentation tool, seems like its pretty cool
"""


def normalize(images, norm_type, scale_0_1=True):
    assert norm_type in constants.norm_types
    if norm_type == constants.per_slice:
        per_slice_norm(images)
    elif norm_type == constants.per_volume:
        per_volume_norm(images)
    else:
        per_dataset_norm(images)


def per_slice_norm(images):
    """
    Args:
    images: list of 3d ndarrays - the dataset
    Normalizes data per slice in each volume:
    disadvantage -> loss of possible relevant features, unique to the slice
    advantage -> removes the intensity differences inside volumes and across volumes
    """
    for image in images:
        for slice in image:
            slice -= np.mean(slice)
            slice /= np.std(slice)


def per_volume_norm(images):
    """
    Normalizes data per volume:
    disadvantage -> loss of possible relevant features, unique to the volume
    advantage -> removes the intensity differences across volumes
    """
    for image in images:
        image -= np.mean(image)
        image /= np.std(image)


def per_dataset_norm(images):
    """
    Normalizes data per whole dataset statistics:
    advantage -> preserves relevant features, unique to the volumes
    disadvantage -> intensity differences across volumes remain
    """
    shady_mean = 0
    for image in images:
        shady_mean += np.mean(image)
    shady_mean /= len(images)

    shady_std = 0
    total_elem = 0
    for image in images:
        total_elem += np.size(image)
        shady_std += np.sum((image - shady_mean) ** 2)
    shady_std = math.sqrt(shady_std / total_elem)

    for image in images:
        image -= shady_mean
        image /= shady_std

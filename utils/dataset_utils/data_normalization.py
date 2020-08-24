import numpy as np
import math

import constants
import general_dataset_settings
"""
- % of images with existing label (not just background) (have to restructure dset)
- mean and std of images - can be done rn (by the looks i must be very careful with normalization)
- should also check out christian payer augmentation tool, seems like its pretty cool
"""


def normalize(images, norm_type, scale_0_1=True):
    assert norm_type in constants.norm_types
    if norm_type == constants.per_slice:
        per_slice_norm(images)


def unnormalize(image, config):
    if config.dataset == constants.imatfib_root_dir:
        mean = general_dataset_settings.imatfib_dataset_mean
        std = general_dataset_settings.imatfib_dataset_std
    elif config.dataset == constants.acdc_root_dir:
        mean = general_dataset_settings.acdc_dataset_mean
        std = general_dataset_settings.acdc_dataset_std

    image = (image * std) + mean
    return image


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


def per_dataset_norm(images):
    """
    Computes mean and std for dataset
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

    return shady_mean, shady_std

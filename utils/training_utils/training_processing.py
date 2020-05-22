import numpy as np
import torch

import constants
from utils import metrics
from utils.ROI_crop import roi_crop


def compute_dice(prediction, mask):
    """
    Args:
    prediction: cuda tensor - model prediction after log softmax (depth, channels, height, width)
    mask: cuda long tensor - ground truth (depth, height, width)

    Returns:
    dice score - ndarray
    concrete prediction - ndarray (depth height width) uint8
    mask - ndarray (depth height width) uint8
    """
    n_classes = prediction.shape[1] - 1
    mask = mask.cpu().numpy().astype(np.uint8)
    # get the maximum values of the channels dimension, then take the indices
    prediction = prediction.max(1)[1]
    prediction = prediction.detach().cpu().numpy().astype(np.uint8)
    dice = metrics.metrics(mask, prediction, n_classes=n_classes)
    return dice, prediction, mask


def process_volume(model, volume, mask, r_info):
    """
    Args:
    model: nn.Module
    volume: cuda tensor (depth height width)
    mask: cuda long tensor - ground truth (depth, height, width)
    r_info: namedtuple - size of original roi and size of original prediction

    Get the processed volume by the model (after log softmax), upsampled to the original size, ie
    the size of the mask

    Returns:
    cuda tensor (depth channels height width)
    """
    processed_volume = []
    for image in volume:
        image = image.view(1, 1, *image.shape)
        processed_volume.append(model(image))
    processed_volume = torch.cat(processed_volume)

    if r_info:
        processed_volume = roi_crop.reinsert_roi(processed_volume, r_info)
    processed_volume = torch.nn.functional.interpolate(processed_volume, mask.shape[1:],
                                                       mode="bilinear")
    return processed_volume

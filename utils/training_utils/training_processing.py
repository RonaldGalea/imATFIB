import numpy as np
import torch

import general_config
from utils import metrics
from utils.ROI_crop import roi_crop
from utils.training_utils import box_utils


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


def process_volume(model, volume, mask, params, r_info=None):
    """
    Args:
    model: nn.Module
    volume: cuda tensor (depth height width) or (depth channels height width)
    mask: cuda long tensor - ground truth (depth, height, width)
    r_info: list - coords of original roi for each slice

    Get the processed volume by the model (after log softmax), upsampled to the original size, ie
    the size of the mask

    Returns:
    cuda tensor (depth channels height width)
    """
    if len(volume.shape) == 3:
        # make batch, channel, height, width
        volume = volume.unsqueeze(1)
    processed_volume = model(volume)

    if r_info:
        processed_volume = roi_crop.reinsert_roi(processed_volume, r_info, params)
    processed_volume = torch.nn.functional.interpolate(processed_volume, mask.shape[1:],
                                                       mode="bilinear")
    return processed_volume


def compute_loc_loss(pred, target, heart_presence, anchor, encompassing_penalty_factor):
    """
    Localization loss is computed similar to https://arxiv.org/abs/1311.2524

    pred - batch x 4 tensor
    target - batch x 4 tensor
    """
    # compute L1 loss
    offsets = box_utils.compute_offsets(target, anchor)
    loc_loss = torch.nn.functional.smooth_l1_loss(pred, offsets, reduction='none')

    loc_loss = loc_loss.sum(dim=1)

    # finally, only take loss out of samples where the heart is present
    # this has dim batch x 1
    return (loc_loss * heart_presence).mean()
    # return torch.tensor(0).to(general_config.device)


def compute_confidence_loss(pred, target, weight):
    """
    pred - batch x 1 tensor, float32
    target - batch tensor
    """

    conf_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred.squeeze(1),
                                                                     target.to(torch.float32),
                                                                     pos_weight=weight,
                                                                     reduction='mean')

    return conf_loss

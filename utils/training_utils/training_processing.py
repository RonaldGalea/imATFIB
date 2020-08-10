import numpy as np
import torch

import general_config
from utils import metrics
from utils.ROI_crop import roi_crop
from utils.training_utils import box_utils
import torchvision.transforms.functional as F


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
    mask = mask.cpu().numpy().astype(np.uint8)
    classes = list(np.unique(mask))
    # do not consider background
    classes.pop(0)
    # get the maximum values of the channels dimension, then take the indices
    prediction = prediction.max(1)[1]
    prediction = prediction.detach().cpu().numpy().astype(np.uint8)
    dice = metrics.metrics(mask, prediction, classes=classes)
    return dice, prediction, mask


def process_volume(model, volume, shape, params, r_info=None, process_in_chunks=False):
    """
    Args:
    model: nn.Module
    volume: cuda tensor (depth height width) or (depth channels height width)
    shape: original shape the ROI must be resized to
    r_info: list - coords of original roi for each slice
    process_in_chunks: - process volume in batches, not as a whole

    Get the processed volume by the model (after log softmax), upsampled to the original size, ie
    the size of the mask

    Returns:
    cuda tensor (depth channels height width)
    """
    if len(volume.shape) == 3:
        # make batch, channel, height, width
        volume = volume.unsqueeze(1)
    if process_in_chunks:
        processed_volume = process_chunks(volume, model, params)
    else:
        processed_volume = model(volume)

    if r_info:
        processed_volume = roi_crop.reinsert_roi(processed_volume, r_info, params)
    processed_volume = torch.nn.functional.interpolate(processed_volume, shape,
                                                       mode="bilinear")
    return processed_volume


def compute_loc_loss(pred, target, heart_presence, gt_for_anc, anchors, encompassing_penalty_factor):
    """
    Localization loss is computed similar to https://arxiv.org/abs/1311.2524

    pred - batch x #anchors x 4 tensor
    target - batch x 4 tensor
    gt_for_anc - batch x #anchors
    anchors - #anchors x 4
    heart_presence - batch
    """
    # compute L1 loss
    # offsets is dim batch x #anchors x 4
    offsets = box_utils.compute_offsets(target, anchors)
    # print("offsets", offsets.shape, offsets)

    loc_loss = torch.nn.functional.smooth_l1_loss(pred, offsets, reduction='none')

    # dimension after this op is batch x #anchors, time to nullify loss for anchors that didn't imatfib_anchor
    loc_loss = loc_loss.sum(dim=2)

    loc_loss = loc_loss * gt_for_anc

    # finally, only take loss out of samples where the heart is present
    # careful to broadcast heart_presence (batch) to (batch x #anchors)
    return (loc_loss * heart_presence.unsqueeze(1)).mean()


def compute_confidence_loss(pred, target, weight):
    """
    pred - batch tensor, float32
    target - batch tensor
    """

    conf_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred,
                                                                     target.to(torch.float32),
                                                                     pos_weight=weight,
                                                                     reduction='mean')

    return conf_loss


def compute_box_conf_loss(roi_conf_pred, gt_for_anc):
    """
    roi_conf_pred: batch x #anchors tensor
    gt_for_anc: batch x #anchors tensor
    """
    box_conf_loss = torch.nn.functional.binary_cross_entropy_with_logits(roi_conf_pred,
                                                                         gt_for_anc,
                                                                         reduction='mean')
    return box_conf_loss


def normalize_volume(volume, mean_std=None):
    """
    Args:
    volume: D x H x W tensor
    mean_std - pair of mean and std values of dataset
    if none, each slice is normalized by its own mean and std
    """
    normalized_image = []
    for slice in volume:
        heigth, width = slice.shape
        slice = slice.view(1, heigth, width)
        if mean_std:
            mean, std = mean_std
            slice = normalize_tensor(slice, [mean, std])
        else:
            slice = normalize_tensor(slice)
        slice = slice.view(heigth, width)
        normalized_image.append(slice)
    image = torch.stack(normalized_image)
    return image


def normalize_tensor(image, mean_std=None):
    """
    Args:
    image: 1 X W x H tensor
    mean_std - pair of mean and std values of dataset
    if none, each slice is normalized by its own mean and std
    """
    if mean_std:
        dataset_mean, dataset_std = mean_std
        image = F.normalize(image, [dataset_mean], [dataset_std])
    else:
        std = torch.std(image)
        if std == 0:
            std = 1
        image = F.normalize(image, [torch.mean(image)], [std])
    return image


def process_chunks(volume, model, params):
    chunks = []
    depth = volume.shape[0]
    cur_slice = 0
    end_slice = min(params.batch_size, depth)
    while True:
        curr_chunk = volume[cur_slice:end_slice]
        print("Current chunk size: ", curr_chunk.shape)
        processed_chunk = process_a_chunk(curr_chunk, model)
        print("Processed chunk device: ", processed_chunk.shape)
        chunks.append(processed_chunk)
        if end_slice == depth:
            return torch.cat(chunks)

        cur_slice = end_slice
        end_slice += params.batch_size
        if end_slice > depth:
            end_slice = depth


def process_a_chunk(chunk, model):
    chunk = chunk.to(general_config.device)
    processed_chunk = model(chunk)
    return processed_chunk.cpu()

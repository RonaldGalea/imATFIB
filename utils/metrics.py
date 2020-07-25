from medpy.metric.binary import dc
import numpy as np
import torch


from utils.training_utils.box_utils import get_intersection_area, get_bbox_area, get_IoU, convert_offsets_to_bboxes, wh2corners


def metrics(img_gt, img_pred, n_classes):
    """
    author: Clément Zotti (clement.zotti@usherbrooke.ca)
    date: April 2017
    - without the volume part

    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    img_pred: np.array
    Array of the predicted segmentation map.

    Return
    ------
    A list of metrics in this order, [Dice LV, Dice RV, Dice MYO]
    """

    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    res = []
    # Loop on each classes of the input images
    for c in range(1, n_classes + 1):
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0

        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)

        # Compute the Dice
        dice = dc(gt_c_i, pred_c_i)

        res += [dice]

    return np.array(res)


def harsh_IOU(pred, target, heart_presence, anchor):
    """
    Regular IOU

    Args:
    pred: batch x 4 tensor
    target: batch x 4 tensor
    """
    # only keep images having a heart instance
    pred = pred[heart_presence.to(torch.bool)]
    target = target[heart_presence.to(torch.bool)]

    # convert model outputs to ((x_ctr, y_ctr, width, height)), then (x_left, y_left, x_right, y_right)
    pred = convert_offsets_to_bboxes(pred, anchor)
    pred = wh2corners(pred)

    intersection_area = get_intersection_area(pred, target)

    iou = get_IoU(pred, target, intersection_area)
    mean_iou = torch.mean(iou)

    return mean_iou


def f1_score(pred, targ):
    """
    Args:
    pred: batch x 1 tensor
    target: batch x 1 tensor
    """
    pred = pred.sigmoid()
    pred[pred > 0.5] = 1

    positives = targ == 1
    tp = pred[positives].sum()
    fp = pred[~positives].sum()
    fn = (1 - pred[positives]).sum()

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    return 2 * (precision * recall) / (precision + recall)

import numpy as np
import cv2
import math
import torch

from utils.dataset_utils import reading
import general_config


def get_dataset_gt_bounds(dataset_name, params):
    """
    Args:
    dataset_name: string - name of the dataset to process

    Get the extreme points of what the ground truths span to form an encompassing square
    ! This depends on the standard height and width, see params_explanation.txt
    """
    split = reading.get_train_val_paths(dataset_name, 5)
    split['train'].extend(split['val'])
    all_samples = split['train']

    max_x, min_x, max_y, min_y = -1, math.inf, -1, math.inf
    for path in all_samples:
        image, mask, _ = reading.get_img_mask_pair(image_path=path, dset_name=dataset_name,
                                                   seg_type=general_config.seg_type)

        # do all computation relative to the actual inference size
        mask = cv2.resize(mask, dsize=(params.default_height, params.default_width))

        c_max_x, c_min_x, c_max_y, c_min_y = get_mask_bounds(mask, params)
        max_x, min_x = max(max_x, c_max_x), min(min_x, c_min_x)
        max_y, min_y = max(max_y, c_max_y), min(min_y, c_min_y)

    print("x max and min: ", max_x, min_x)
    print("y max and min: ", max_y, min_y)

    return max_x, min_x, max_y, min_y


def get_mask_bounds(mask, params):
    # get indeces of label
    label_coords = np.argwhere(mask > 0)
    # no label, just get the image as a whole
    if label_coords.size == 0:
        return params.default_width - 1, 0, params.default_height - 1, 0
    xs, ys = label_coords[:, 1], label_coords[:, 0]

    return np.max(xs), np.min(xs), np.max(ys), np.min(ys)


def reinsert_roi(prediction, reconstruction_info, params):
    """
    Args:
    prediction: torch.tensor - model prediction (depth, roi_height, roi_width)

    Simulates reinserting the ROI in the prediction
    Since there is nothing else predicted apart from ROI, just place it in an empty tensor, at
    the correct place

    Since the roi_height and roi_width are likely different than the actual roi size (since these
    have to be made divisible by the model os), there are some steps before reinsertion:

    1. create empty tensor of original size
    2. resize prediction to original roi size, for each prediction
    3. insert
    """
    # reconstruct original shape
    depth, n_classes = prediction.shape[:2]
    height, width = params.default_height, params.default_width
    original_pred = torch.zeros((depth, n_classes, height, width))
    original_pred = original_pred.to(general_config.device)

    # resize predicted rois
    orig_roi_sizes = reconstruction_info
    for idx, (x_max, x_min, y_max, y_min) in enumerate(orig_roi_sizes):
        # print("In roi crop: ", idx, x_max, x_min, y_max, y_min)
        orig_roi_width, orig_roi_height = x_max - x_min + 1, y_max - y_min + 1
        # simulate mini batch of 1 for interpolate function
        current_roi = prediction[idx].unsqueeze(0)
        original_roi = torch.nn.functional.interpolate(current_roi,
                                                       (orig_roi_height, orig_roi_width),
                                                       mode="bilinear")
        # insert back
        original_pred[idx, :, y_min:y_max+1, x_min:x_max+1] = original_roi
    return original_pred

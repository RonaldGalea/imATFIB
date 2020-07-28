import numpy as np
import cv2
import math
import torch

import constants
from utils.dataset_utils import reading
import general_config
from utils import visualization


def get_dataset_gt_bounds(dataset_name, params, config):
    """
    Args:
    dataset_name: string - name of the dataset to process

    Get the extreme points of what the ground truths span to form an encompassing square
    ! This depends on the standard height and width, see params_explanation.txt
    """
    split = reading.get_train_val_paths(dataset_name, 5)
    split['train'].extend(split['val'])
    all_samples = split['train']

    x_max, x_min, y_max, y_min = -1, math.inf, -1, math.inf
    for path in all_samples:
        image, mask, _ = reading.get_img_mask_pair(image_path=path, dset_name=dataset_name,
                                                   seg_type=config.seg_type)

        # do all computation relative to the actual inference size
        mask = cv2.resize(mask, dsize=(params.default_height, params.default_width))

        c_x_min, c_y_min, c_x_max, c_y_max = get_mask_bounds(mask, params)
        x_max, x_min = max(x_max, c_x_max), min(x_min, c_x_min)
        y_max, y_min = max(y_max, c_y_max), min(y_min, c_y_min)

    print("x max and min: ", x_max, x_min)
    print("y max and min: ", y_max, y_min)

    return x_min, y_min, x_max, y_max


def get_mask_bounds(mask, params):
    # get indeces of label
    label_coords = np.argwhere(mask > 0)
    # no label, just get the image as a whole
    if label_coords.size == 0:
        return 0, 0, params.default_width - 1, params.default_height - 1
    xs, ys = label_coords[:, 1], label_coords[:, 0]

    x_max, x_min = np.max(xs), np.min(xs)
    y_max, y_min = np.max(ys), np.min(ys)
    return x_min, y_min, x_max, y_max


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
    for idx, (x_min, y_min, x_max, y_max) in enumerate(orig_roi_sizes):
        # print("In roi crop: ", idx, x_min, y_min, x_max, y_max)
        orig_roi_width, orig_roi_height = x_max - x_min + 1, y_max - y_min + 1
        # simulate mini batch of 1 for interpolate function
        current_roi = prediction[idx].unsqueeze(0)
        original_roi = torch.nn.functional.interpolate(current_roi,
                                                       (orig_roi_height, orig_roi_width),
                                                       mode="bilinear")
        # insert back
        original_pred[idx, :, y_min:y_max+1, x_min:x_max+1] = original_roi
    return original_pred


def extract_ROI(image, box_coords):
    """
    Args:
    image: ndarray: 2D sample image from which to extract roi

    Extracts a regtangle part of the image, containing the area of interest (labeled area)
    """
    x_min, y_min, x_max, y_max = box_coords

    # extract the roi, relative or absolute
    roi_horizontal = slice(x_min, x_max+1)
    roi_vertical = slice(y_min, y_max+1)

    roi = image[roi_vertical, roi_horizontal]

    return roi


def get_minimum_size(x_min, y_min, x_max, y_max, params):
    """
    In the case of relative roi extraction, resize crop such that the minimum size is
    default height/width // 4
    """
    width = params.default_width
    height = params.default_height

    if x_max - x_min + 1 < width // 4:
        mid = (x_max + x_min) // 2
        x_max = mid + width // 8
        x_min = mid - (width // 8) - 1

    if y_max - y_min + 1 < height // 4:
        mid = (y_max + y_min) // 2
        y_max = mid + height // 8
        y_min = mid - (height // 8) - 1

    x_min, y_min, x_max, y_max = clamp_values(x_min, y_min, x_max, y_max, width, height)

    return x_min, y_min, x_max, y_max


def extract_ROI_3d(volume, mask, params, config):
    roi_volume = []
    orig_roi_infos = []
    for i, (image_slice, mask_slice) in enumerate(zip(volume, mask)):
        box_coords = compute_ROI_coords(mask_slice, params, config, validation=True)
        roi = extract_ROI(image_slice, box_coords)
        orig_roi_infos.append(box_coords)

        # resize roi to input res
        roi = cv2.resize(roi, dsize=(params.roi_width, params.roi_height))
        roi_volume.append(roi)

    roi_volume = np.stack(roi_volume)

    return roi_volume, orig_roi_infos


def extract_ROI_from_pred(volume, params, predicted_coordinates):
    roi_volume = []
    orig_roi_infos = []

    for i, (image_slice, coords) in enumerate(zip(volume, predicted_coordinates)):
        roi = extract_ROI(image_slice.squeeze(0), coords)
        orig_roi_infos.append(coords)

        # resize roi to input res
        roi = roi.reshape(1, 1, *roi.shape)
        roi = torch.nn.functional.interpolate(
            roi, (params.roi_height, params.roi_width), mode="bilinear")
        roi_volume.append(roi)
    roi_volume = torch.cat(roi_volume, dim=0)

    return roi_volume, orig_roi_infos


def compute_ROI_coords(mask, params, config, validation=False):
    """
    mask: ndarray: 2D label image, used for extracting relativ roi coords
    validation: bool: relatvie roi extraction differs

    computes bounds of labelled area

    returns: tuple of box coords
    """
    x_min, y_min, x_max, y_max = get_mask_bounds(mask, params)
    x_min, y_min, x_max, y_max = get_minimum_size(x_min, y_min, x_max, y_max, params)

    if config.model_id in constants.segmentor_ids:
        if params.relative_roi_perturbation:
            x_min, y_min, x_max, y_max = add_perturbation(
                x_min, y_min, x_max, y_max, params, validation)

    elif config.model_id in constants.detectors:
        # margin should be added before minimum size
        x_min, y_min, x_max, y_max = add_detection_error_margin(
            x_min, y_min, x_max, y_max, params)

    return (x_min, y_min, x_max, y_max)


def add_perturbation(x_min, y_min, x_max, y_max, params, validation=False):
    # practically, region cut won't always be perfect, so add a perturbation value
    perfect_roi_width, perfect_roi_height = x_max - x_min, y_max - y_min
    width_perturb_limit = perfect_roi_width // 5
    height_perturb_limit = perfect_roi_height // 5
    min_width_perturb = perfect_roi_width // 20
    min_height_perturb = perfect_roi_height // 20

    # perturbation (error) for detector will be exactly // 10

    if not validation:
        # perturb up to 33% of original size
        x_min_perturb = np.random.randint(min_width_perturb, width_perturb_limit+1)
        x_max_perturb = np.random.randint(min_width_perturb, width_perturb_limit+1)
        y_max_perturb = np.random.randint(min_height_perturb, height_perturb_limit+1)
        y_min_perturb = np.random.randint(min_height_perturb, height_perturb_limit+1)
    else:
        # if we're validating, add fixed perturbation to avoid a lucky eval
        x_min_perturb = width_perturb_limit // 2
        x_max_perturb = width_perturb_limit // 2
        y_max_perturb = height_perturb_limit // 2
        y_min_perturb = height_perturb_limit // 2

    x_min -= x_min_perturb
    x_max += x_max_perturb
    y_min -= y_min_perturb
    y_max += y_max_perturb

    # clamp values back to image range
    width = params.default_width
    height = params.default_height

    x_min, y_min, x_max, y_max = clamp_values(x_min, y_min, x_max, y_max, width, height)
    return x_min, y_min, x_max, y_max


def add_detection_error_margin(x_min, y_min, x_max, y_max, params):
    perfect_roi_width, perfect_roi_height = x_max - x_min, y_max - y_min
    width_perturb = perfect_roi_width // 10
    height_perturb = perfect_roi_height // 10

    x_min -= width_perturb
    x_max += width_perturb
    y_min -= height_perturb
    y_max += height_perturb

    # clamp values back to image range
    width = params.default_width
    height = params.default_height

    x_min, y_min, x_max, y_max = clamp_values(x_min, y_min, x_max, y_max, width, height)
    return x_min, y_min, x_max, y_max


def clamp_values(x_min, y_min, x_max, y_max, width, height):
    x_min = max(x_min, 0)
    x_max = min(x_max, width - 1)
    y_min = max(y_min, 0)
    y_max = min(y_max, height - 1)

    return x_min, y_min, x_max, y_max

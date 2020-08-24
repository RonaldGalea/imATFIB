import numpy as np
import cv2
import math
import torch

import constants
from utils.dataset_utils import reading


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
        image, mask, _ = reading.read_img_mask_pair(image_path=path, dset_name=dataset_name,
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
    original_pred = original_pred.to(prediction.device)

    # resize predicted rois
    orig_roi_sizes = reconstruction_info
    for idx, (x_min, y_min, x_max, y_max) in enumerate(orig_roi_sizes):
        orig_roi_width, orig_roi_height = x_max - x_min + 1, y_max - y_min + 1

        # simulate mini batch of 1 for interpolate function
        current_roi = prediction[idx].unsqueeze(0)
        original_roi = torch.nn.functional.interpolate(current_roi,
                                                       (orig_roi_height, orig_roi_width),
                                                       mode="bilinear", align_corners=False)
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


def extract_ROI_from_pred(volume, params, predicted_coordinates):
    """
    volume: D x H x W ndarray
    predicted coordinates: list of roi coords for the slices of the volume

    returns:
    returns - D x H x W roi ndarray
    """
    if torch.is_tensor(volume):
        volume = volume.detach().cpu().numpy()
    roi_volume = []
    for i, (image_slice, coords) in enumerate(zip(volume, predicted_coordinates)):
        roi = extract_ROI(image_slice, coords)

        # resize roi to input res
        roi = cv2.resize(roi, dsize=(params.roi_width, params.roi_height))
        roi_volume.append(roi)

    roi_volume = np.stack(roi_volume)

    return roi_volume


def extract_ROI_3d(volume, mask, params, config):
    """
    volume - D x H x W ndarray

    returns - D x H x W roi ndarray
    """
    roi_volume = []
    orig_roi_infos = []
    coords_n_scores = get_volume_coords(mask, params, config, validation=True)
    for i, (image_slice, _, coords_n_score) in enumerate(zip(volume, mask, coords_n_scores)):
        box_coords = coords_n_score[:4]
        roi = extract_ROI(image_slice, box_coords)
        orig_roi_infos.append(box_coords)

        # resize roi to input res
        roi = cv2.resize(roi, dsize=(params.roi_width, params.roi_height))
        roi_volume.append(roi)

    roi_volume = np.stack(roi_volume)

    return roi_volume, orig_roi_infos


def get_volume_coords(mask, params, config, validation=False, double_seg=False):
    """
    Iterates the volume slices and computes a bounding box for the labelled area
    For unlabelled slices, a flag will be set
    """
    if double_seg:
        setup = [False, True, True]
    else:
        setup = get_roi_crop_setup(params, config)
    coords_n_scores = []
    if torch.is_tensor(mask):
        mask = mask.numpy()
    for slice in mask:
        (x_min, y_min, x_max, y_max) = compute_ROI_coords(
            slice, params, setup, validation=validation)
        score = no_roi_check(x_min, y_min, x_max, y_max, params)
        coords_n_scores.append((x_min, y_min, x_max, y_max, score))
    return coords_n_scores


def compute_ROI_coords(mask, params, setup, validation=False):
    """
    mask: ndarray: 2D label image, used for extracting relativ roi coords
    validation: bool: relatvie roi extraction differs

    computes bounds of labelled area

    returns: tuple of box coords
    """
    perturbation, detection_err_margin, use_min_size = setup
    x_min, y_min, x_max, y_max = get_mask_bounds(mask, params)

    if perturbation:
        x_min, y_min, x_max, y_max = add_perturbation(x_min, y_min, x_max, y_max,
                                                      params, validation)

    if detection_err_margin:
        x_min, y_min, x_max, y_max = add_detection_error_margin(x_min, y_min, x_max, y_max, params)

    if use_min_size:
        x_min, y_min, x_max, y_max = get_minimum_size(x_min, y_min, x_max, y_max, params)

    return (x_min, y_min, x_max, y_max)


def add_perturbation(x_min, y_min, x_max, y_max, params, validation=False):
    # practically, region cut won't always be perfect, so add a perturbation value
    perfect_roi_width, perfect_roi_height = x_max - x_min, y_max - y_min
    max_perturb, min_perturb = params.relative_roi_perturbation

    width_perturb_limit = int(perfect_roi_width / max_perturb)
    height_perturb_limit = int(perfect_roi_height / max_perturb)
    min_width_perturb = int(perfect_roi_width / min_perturb)
    min_height_perturb = int(perfect_roi_height / min_perturb)

    if not validation:
        # perturb up to x% of original size
        x_min_perturb = np.random.randint(min_width_perturb, width_perturb_limit+1)
        x_max_perturb = np.random.randint(min_width_perturb, width_perturb_limit+1)
        y_max_perturb = np.random.randint(min_height_perturb, height_perturb_limit+1)
        y_min_perturb = np.random.randint(min_height_perturb, height_perturb_limit+1)
    else:
        # if we're validating, add fixed perturbation to avoid a lucky eval
        x_min_perturb = int(width_perturb_limit / 2)
        x_max_perturb = int(width_perturb_limit / 2)
        y_max_perturb = int(height_perturb_limit / 2)
        y_min_perturb = int(height_perturb_limit / 2)

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
    err_margin = params.err_margin
    width_perturb = int(perfect_roi_width / err_margin)
    height_perturb = int(perfect_roi_height / err_margin)

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


def get_roi_crop_setup(params, config):
    perturb = params.relative_roi_perturbation
    err_margin = params.use_det_err_margin if config.model_id in constants.detectors else False
    use_min_size = params.use_min_size
    setup = [perturb, err_margin, use_min_size]
    return setup


def no_roi_check(x_min, y_min, x_max, y_max, params):
    if (0, 0, params.default_width - 1, params.default_height - 1) == (x_min, y_min, x_max, y_max):
        return 0
    return 1

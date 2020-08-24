import torch
import numpy as np

import general_config
from utils.training_utils.box_utils import convert_offsets_to_bboxes, wh2corners
from utils.training_utils import training_processing
from utils.ROI_crop import roi_crop


def scale_coords(coords_n_score, height, width, down=True):
    """
    coords_n_score: batch x 5 (coords xyxy + score)
    width, height: scaling factors
    down: scale down if true
    """
    if not down:
        width = 1 / width
        height = 1 / height
    coords_n_score[:, 0] = coords_n_score[:, 0] / width
    coords_n_score[:, 1] = coords_n_score[:, 1] / height
    coords_n_score[:, 2] = coords_n_score[:, 2] / width
    coords_n_score[:, 3] = coords_n_score[:, 3] / height


def get_segmentor_pred_coords(validation_loader, segmentor_base, params, config):
    """
    Returns a list of ndarrays of size D x 5, representing ROI coordinates and heart presence flag
    of each slice for each volume
    """
    segmentor_base.eval()

    coords_n_scores = []
    with torch.no_grad():
        for batch_nr, (volume, _, _, _) in enumerate(validation_loader):
            volume = volume.to(general_config.device)

            processed_volume = training_processing.process_volume(segmentor_base, volume,
                                                                  volume.shape[1:], params)
            prediction = processed_volume.max(1)[1]
            prediction = prediction.detach().cpu().numpy().astype(np.int32)

            coords_n_score = roi_crop.get_volume_coords(prediction, params, config, double_seg=True)

            coords_n_score = np.array(coords_n_score).astype(np.float32)
            # normalize coords first so they are usable at any dimensions
            height, width = prediction.shape[-2:]
            scale_coords(coords_n_score, height, width)
            coords_n_scores.append(coords_n_score)

    return coords_n_scores


def segment_with_computed_ROI(validation_loader, segmentor, params, config, coords_n_scores, logsoft_preds=False):
    """
    use a ROI trained model to segment from computed ROI coords
    logsoft_preds: bool, returns log softmax prediction if true
    """
    segmentor.eval()
    validation_loader.dataset.visualization_mode = True

    total_dice, predicted_volumes = 0, []
    with torch.no_grad():
        for batch_nr, ((_, mask, _, volume), cns) in enumerate(zip(validation_loader, coords_n_scores)):
            roi_volume, coords, scores = get_roi_from_model(cns, params, volume)

            # process ROI with segementor
            processed_volume = training_processing.process_volume(segmentor, roi_volume,
                                                                  mask.shape[1:],
                                                                  params, list(coords),
                                                                  process_in_chunks=True)

            # overwrite indices of no heart scores with all background predictions
            processed_volume = no_heart_means_background(processed_volume, scores)
            dice, final_pred, mask_npy = training_processing.compute_dice(processed_volume, mask)

            print("current volume dice: ", dice, np.mean(dice))
            total_dice += dice
            if logsoft_preds:
                predicted_volumes.append(processed_volume)
            else:
                predicted_volumes.append(final_pred)

    mean_val_dice = total_dice / len(validation_loader)
    print("Mean dice: ", mean_val_dice, np.mean(mean_val_dice))
    print("Total volumes: ", len(validation_loader))

    return total_dice / len(validation_loader), predicted_volumes


def segment_with_computed_ROI_test(test_loader, segmentor, params, config, coords_n_scores, logsoft_preds=False):
    segmentor.eval()

    predicted_volumes = []
    with torch.no_grad():
        for batch_nr, ((volume, orig_shape, header_info, _), cns) in enumerate(zip(test_loader, coords_n_scores)):
            roi_volume, coords, scores = get_roi_from_model(cns, params, volume)

            # process ROI with segementor
            processed_volume = training_processing.process_volume(segmentor, roi_volume, orig_shape,
                                                                  params, list(coords),
                                                                  process_in_chunks=True)

            # overwrite indices of no heart scores with all background predictions
            processed_volume = no_heart_means_background(processed_volume, scores)
            prediction = processed_volume.max(1)[1]
            prediction = prediction.detach().cpu().numpy().astype(np.uint8)

            if logsoft_preds:
                predicted_volumes.append(processed_volume)
            else:
                predicted_volumes.append(prediction)

    return predicted_volumes


def get_roi_from_model(cns, params, volume):
    height, width = volume.shape[-2:]
    scale_coords(cns, height, width, down=False)

    coords = cns[:, :4].astype(np.int32)
    scores = cns[:, 4]

    # extract ROI with detector preds
    roi_volume = roi_crop.extract_ROI_from_pred(volume, params, coords)
    roi_volume = torch.tensor(roi_volume)

    # normalize roi volume
    roi_volume = training_processing.normalize_volume(roi_volume)
    roi_volume = roi_volume.unsqueeze(1)

    return roi_volume, coords, scores


def no_heart_means_background(processed_volume, scores):
    background = torch.zeros(processed_volume.shape[1:]).to(processed_volume.device)
    # simulate softmax probab max for background index
    background[0, :, :] = 1

    # insert this tensor in background predicted indices
    # print("IN NO HEART MEANDS BACK\n\n\n", scores)
    processed_volume[scores < 0.5] = background
    return processed_volume


def get_detector_output(image, detector, params):
    ROI_pred, ROI_conf_pred, score_pred = detector(image)
    most_conf = ROI_conf_pred.argmax(dim=1)
    confident_ROI_pred = ROI_pred[torch.arange(len(most_conf)), most_conf]
    anchors = detector.anchors_xyxy[most_conf]
    pred = convert_offsets_to_bboxes(confident_ROI_pred, anchors)
    pred = wh2corners(pred) * params.default_height

    # clamp predicted coords in image bounds
    pred = torch.clamp(pred, 0, params.default_height)
    pred = pred.to(torch.int32).cpu().numpy()

    score_pred = score_pred.sigmoid()
    return pred, score_pred

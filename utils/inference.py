import torch

import general_config
import constants
import numpy as np
from utils.training_utils import training_processing


def run_model_inference(dataloader, model, params, logsoft_preds=False):
    """
    Args:
    dataloader: validation dataloader, has to return one volume at a time
    model: segmentation model
    params of that model
    logsoft_preds: bool, returns log softmax prediction if true

    Returns:
    final_preds: a list of volumes (D x H x W ndarrays)
    all_dices: a list of lists of dice values for each slices of the volumes
    """
    dataloader.dataset.visualization_mode = True
    final_preds, all_dices = [], []
    for vol_idx, (volume, mask, r_infos, orig_volume) in enumerate(dataloader):
        volume, mask = volume.to(general_config.device), mask.to(general_config.device)

        # process volume slice by slice so we can see dice per each one
        dices, pred_slices = [], []
        for slice_idx, (vol_slice, mask_slice) in enumerate(zip(volume, mask)):
            # simulate bs = 1
            vol_slice = vol_slice.unsqueeze(0)
            mask_slice = mask_slice.unsqueeze(0)

            r_info = None if params.roi_crop == constants.no_roi_extraction else [
                r_infos[slice_idx]]

            # this has softmax channels, (cuda) tensor
            processed_slice = training_processing.process_volume(model, vol_slice,
                                                                 mask_slice.shape[1:],
                                                                 params, r_info)
            # this is final pred, cpu, np.uint8
            slice_dice, final_slice, _ = training_processing.compute_dice(processed_slice,
                                                                          mask_slice)
            dices.append(slice_dice)
            if logsoft_preds:
                pred_slices.append(processed_slice)
            else:
                pred_slices.append(final_slice)

        # save current model prediction and stats
        pred_volume = np.concatenate(pred_slices)
        final_preds.append(pred_volume)
        all_dices.append(dices)

    return final_preds, all_dices


def ensemble_inference(logsoftmax_volumes, validation_dataloader):
    """
    Args:
    logsoftmax_volumes: list of lists representing the predictions of each model for each volume
    validation_dataloader: just to get the masks for metrics computation
    """
    n_models, n_volumes = len(logsoftmax_volumes), len(validation_dataloader)

    volumes_pred = [[None for i in range(n_models)] for j in range(n_volumes)]

    for i in range(n_volumes):
        for j in range(n_models):
            volumes_pred[i][j] = logsoftmax_volumes[j][i]

    total_dice, final_combined_preds = 0, []
    for batch_nr, (_, mask, _, _) in enumerate(validation_dataloader):
        current_vol_predictions = volumes_pred[batch_nr]
        combined = combine_predictions(current_vol_predictions)
        dice, final_combined, _ = training_processing.compute_dice(combined, mask)
        total_dice += dice
        final_combined_preds.append(combined)

    print("Final validation results: ", total_dice / len(validation_dataloader))
    return final_combined_preds


def combine_predictions(processed_volumes):
    """
    processed_volumes - list of Predictions after log softmax of each model on the current volume
    """
    out = torch.zeros(processed_volumes[0].shape)
    for vol in processed_volumes:
        out += torch.exp(vol)
    return out / len(processed_volumes)

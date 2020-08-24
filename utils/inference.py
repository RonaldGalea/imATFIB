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
    with torch.no_grad():
        for vol_idx, (volume, mask, r_infos, _) in enumerate(dataloader):
            volume, mask = volume.to(general_config.device), mask.to(general_config.device)

            # process volume slice by slice so we can see dice per each one
            dices, pred_slices = [], []
            for slice_idx, (vol_slice, mask_slice) in enumerate(zip(volume, mask)):
                # simulate bs = 1
                vol_slice = vol_slice.unsqueeze(0)
                mask_slice = mask_slice.unsqueeze(0)

                r_info = [r_infos[slice_idx]] if hasattr(params, "roi_crop") else None

                # this has softmax channels, (cuda) tensor
                processed_slice = training_processing.process_volume(model, vol_slice,
                                                                     mask_slice.shape[1:],
                                                                     params, r_info)
                # this is final pred, cpu, np.uint8
                slice_dice, final_slice, _ = training_processing.compute_dice(processed_slice,
                                                                              mask_slice)
                dices.append(slice_dice)
                if logsoft_preds:
                    pred_slices.append(processed_slice.cpu().numpy())
                else:
                    pred_slices.append(final_slice)

            # save current model prediction and stats
            pred_volume = np.concatenate(pred_slices)
            final_preds.append(pred_volume)
            all_dices.append(dices)

    return final_preds, all_dices


def run_model_test_inference(dataloader, model, params, logsoft_preds=False):
    """
    Args:
    dataloader: validation dataloader, has to return one volume at a time
    model: segmentation model
    params of that model
    logsoft_preds: bool, returns log softmax prediction if true

    Returns:
    final_preds: a list of volumes (D x H x W ndarrays)
    """
    final_preds = []
    with torch.no_grad():
        for vol_idx, (volume, orig_shape, header_info, _) in enumerate(dataloader):
            processed_volume = training_processing.process_volume(model, volume, orig_shape,
                                                                  params, process_in_chunks=True)
            if logsoft_preds:
                final_preds.append(processed_volume)
            else:
                prediction = processed_volume.max(1)[1]
                prediction = prediction.detach().cpu().numpy().astype(np.uint8)
                final_preds.append(prediction)

    return final_preds


def ensemble_inference(logsoftmax_volumes, validation_dataloader):
    """
    Args:
    logsoftmax_volumes: list of lists representing the predictions of each model for each volume
    validation_dataloader: just to get the masks for metrics computation
    """
    volumes_pred = reorder_volumes(logsoftmax_volumes)
    total_dice, final_combined_preds = 0, []
    for batch_nr, (_, mask, _, _) in enumerate(validation_dataloader):
        current_vol_predictions = volumes_pred[batch_nr]
        combined = combine_predictions(current_vol_predictions)
        dice, final_combined, _ = training_processing.compute_dice(combined, mask)
        print("Current volume dice: ", dice)
        total_dice += dice
        final_combined_preds.append(final_combined)

    mean_val_dice = total_dice / len(validation_dataloader)
    print("Final validation results: ", mean_val_dice, np.mean(mean_val_dice))
    return final_combined_preds


def ensemble_inference_test(logsoftmax_volumes):
    volumes_pred = reorder_volumes(logsoftmax_volumes)
    final_combined_preds = []
    for c_volumes in volumes_pred:
        combined = combine_predictions(c_volumes)
        prediction = combined.max(1)[1]
        prediction = prediction.detach().cpu().numpy().astype(np.uint8)
        final_combined_preds.append(prediction)
    return final_combined_preds


def combine_predictions(processed_volumes):
    """
    processed_volumes - list of Predictions after log softmax of each model on the current volume
    """
    out = torch.zeros(processed_volumes[0].shape)
    for vol in processed_volumes:
        softmax = torch.exp(vol)
        softmax = softmax.cpu()
        out += softmax
    return out / len(processed_volumes)


def reorder_volumes(logsoftmax_volumes):
    n_models, n_volumes = len(logsoftmax_volumes), len(logsoftmax_volumes[0])

    volumes_pred = [[None for i in range(n_models)] for j in range(n_volumes)]

    for i in range(n_volumes):
        for j in range(n_models):
            if not torch.is_tensor(logsoftmax_volumes[j][i]):
                volumes_pred[i][j] = torch.from_numpy(logsoftmax_volumes[j][i])
            else:
                volumes_pred[i][j] = logsoftmax_volumes[j][i]
    return volumes_pred

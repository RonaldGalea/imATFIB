import torch
import numpy as np
from pathlib import Path

import constants
import general_config
import ACDC_metrics
from utils.training_utils import training_processing
from utils import prepare_models
from utils.training_utils import training_setup
from utils.params import Params


def process_and_save(model_id, dataset_name):
    """
    model_id - string
    dataset_name - string
    """
    model, params, config = prepare_models.prepare([model_id], dataset_name)
    validation_dataloader = training_setup.prepare_val_loader(params, config)

    with torch.no_grad():
        for batch_idx, (volume, mask, r_info, header_info) in enumerate(validation_dataloader):
            print("Original shape: ", mask.shape)
            mask = mask.numpy().astype(np.uint8)

            process_and_save_volume(volume, model, params, mask.shape[1:], header_info, batch_idx)
            save_volume(mask, header_info, batch_idx, type="gt")

            print("Volume number ", batch_idx+1, " saved successfully!")


def save_predictions(predicted_volumes, model_id, dataset_name):
    """
    list of ndarrays representing volume predictions
    """
    config = Params(constants.config_path.format(dataset_name, model_id))
    params = Params(constants.params_path.format(dataset_name, model_id))
    validation_dataloader = training_setup.prepare_val_loader(params, config)

    for batch_idx, (_, mask, _, header_info) in enumerate(validation_dataloader):
        mask = mask.numpy().astype(np.uint8)
        processed_volume = predicted_volumes[batch_idx]
        save_volume(processed_volume, header_info, batch_idx)
        save_volume(mask, header_info, batch_idx, type="gt")

        print("Volume number ", batch_idx+1, " saved successfully!")


def process_and_save_test(model_id, dataset_name, chunks=False):
    """
    """
    model, params, config = prepare_models.prepare([model_id], dataset_name)
    test_dataloader = training_setup.prepare_test_loader(params, config)

    with torch.no_grad():
        for batch_idx, (volume, orig_shape, header_info, img_path) in enumerate(test_dataloader):
            process_and_save_volume(volume, model, params, orig_shape,
                                    header_info, batch_idx, img_path=img_path, chunks=chunks)


def save_predictions_test(predicted_volumes, model_id, dataset_name):
    """
    list of ndarrays representing volume predictions
    """
    config = Params(constants.config_path.format(dataset_name, model_id))
    params = Params(constants.params_path.format(dataset_name, model_id))
    test_dataloader = training_setup.prepare_test_loader(params, config)

    for batch_idx, (volume, orig_shape, header_info, img_path) in enumerate(test_dataloader):
        print("Volume shape: ", volume.shape, " and original shape: ", orig_shape)
        processed_volume = predicted_volumes[batch_idx]
        save_volume(processed_volume, header_info, batch_idx, img_path=img_path)
        print("Volume number ", batch_idx+1, " saved successfully!")


def process_and_save_volume(volume, model, params, orig_shape, header_info, batch_idx,
                            r_info=None, img_path=None, chunks=False):
    print("Volume shape: ", volume.shape, " and original shape: ", orig_shape)
    if not chunks:
        volume = volume.to(general_config.device)
    processed_volume = training_processing.process_volume(model, volume, orig_shape,
                                                          params, r_info, process_in_chunks=chunks)
    processed_volume = processed_volume.max(1)[1]
    processed_volume = processed_volume.detach().cpu().numpy().astype(np.uint8)
    save_volume(processed_volume, header_info, batch_idx, img_path=img_path)


def save_volume(volume, header_info, batch_idx, type="prediction", img_path=None):
    if img_path:
        name_no_ext = str(Path(img_path.stem).stem)
        if "patient" in name_no_ext:
            path = "acdc_test/"
            patient, frame = name_no_ext.split('_')
            frame_nr = frame[-2:]
            mri_type = "ES"
            if frame_nr == "01":
                mri_type = "ED"
            path += patient + "_" + mri_type + ".nii.gz"
        else:
            path = "mmwhs_test/"
            save_name = name_no_ext.split("_encrypt_")[0]
            path += save_name + ".nii.gz"
    else:
        dir = "volumes/"
        if type == "gt":
            dir = "gt_volumes/"
        path = dir + str(batch_idx) + ".nii.gz"

    affine, header = header_info
    volume = np.transpose(volume, (1, 2, 0))
    ACDC_metrics.save_nii(path, volume, affine, header)
    print("Saved volume shape, dtype and unique elems: ", volume.shape, volume.dtype, np.unique(volume))

    if img_path:
        print("Volume path: ", img_path)

import numpy as np


import general_config
import ACDC_metrics
from utils.training_utils import training_processing
from utils import prepare_models_and_data


def process_and_save(model_id, dataset_name):
    """
    """
    models, validation_dataloader, params = prepare_models_and_data.prepare(model_id, dataset_name)
    model = models[0]
    for batch_idx, (volume, mask, r_info, header_info) in enumerate(validation_dataloader):
        volume, mask = volume.to(general_config.device), mask.to(general_config.device)
        processed_volume = training_processing.process_volume(model, volume, mask,
                                                              params, r_info)
        mask = mask.cpu().numpy().astype(np.uint8)
        processed_volume = processed_volume.max(1)[1]
        processed_volume = processed_volume.detach().cpu().numpy().astype(np.uint8)
        affine, header = header_info
        path = "volumes/" + str(batch_idx) + ".nii"
        gts_path = "gt_volumes/" + str(batch_idx) + ".nii"
        ACDC_metrics.save_nii(path, processed_volume, affine, header)
        ACDC_metrics.save_nii(gts_path, mask, affine, header)
        print("Volume number ", batch_idx+1, " saved successfully!")

"""
Prepare models for visualization of results or ensembling
"""
import torch

import constants
import general_config
from utils.params import Params, validate_params
from utils.training_utils import training_setup, prints, training_processing
from training import model_statistics


def prepare(model_ids, dataset_name):
    """
    Args:
    model_ids: list of str - ids of models for inference
    dataset_name: str - will construct validation loader of the dataset

    ! Important:
    models are expected to have same settings regarding dataset and training (eg roi, input res)

    Returns:
    loaded models ready for inference in a list
    validation_loader
    params
    """
    models, params_list, configs_list = [], [], []
    for id in model_ids:
        params = Params(constants.params_path.format(dataset_name, id))
        config = Params(constants.config_path.format(dataset_name, id))
        validate_params(params)
        print("Constructing model: ", id)
        model = training_setup.model_setup(dataset_name, params, config)
        # in this case, model id is equal to exp name
        training_setup.load_model_weights(model, dataset_name, id)
        model.eval()
        prints.print_trained_parameters_count(model)
        models.append(model)
        params_list.append(params)
        configs_list.append(config)

    return models, params_list, configs_list


def ensemble_inference(model_ids, dataset_name):
    models, validation_dataloader, params = prepare(model_ids, dataset_name)
    val_statistics = model_statistics.Model_Statistics(len(validation_dataloader),
                                                       params, models[0].n_classes - 1,
                                                       'val')
    loss_function = torch.nn.NLLLoss(weight=None, reduction='mean')
    with torch.no_grad():
        for batch_nr, (image, mask, r_info) in enumerate(validation_dataloader):
            image, mask = image.to(general_config.device), mask.to(general_config.device)
            # process the input with each model
            processed_volumes = []
            for model in models:
                processed_volume = training_processing.process_volume(
                    model, image, mask, params, r_info)
                processed_volumes.append(processed_volume)
            combined = combine_predictions(processed_volumes)
            loss = loss_function(combined, mask)
            dice, _, _ = training_processing.compute_dice(combined, mask)
            val_statistics.update(loss.item(), dice)

    print("Final validation results: ")
    val_statistics.print_batches_statistics()


def combine_predictions(processed_volumes):
    out = torch.zeros(processed_volumes[0].shape)
    for vol in processed_volumes:
        out += torch.exp(vol)
    return out / len(processed_volumes)

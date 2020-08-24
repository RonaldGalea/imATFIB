try:
    from apex import amp
    amp_available = True
except ImportError:
    print("Cannot import NVIDIA Apex...")

import general_config
import constants
from training import train
from utils.dataset_utils import data_normalization
from utils.params import Params, validate_params
from utils.training_utils import prints, training_setup
from utils.ROI_crop import roi_crop


def main(dataset_name, experiment_name, train_model=False, evaluate_model=False,
         compute_dset_mean_std=False, compute_dset_gt_bounds=False):
    """
    Args:
    dataset_name: str - name of the dataset -> from constants.py
    experiment_name: str - name of the experiment folder
    train_model: bool - begin trainig if true
    evaluate_model: bool - run validation if true
    compute_dset_mean_std: bool - compute mean and std of the dataset
    compute_dset_gt_bounds: bool - compute coordinates of an encompassing box of the labelled area
    of the training set
    """
    config = Params(constants.config_path.format(dataset_name, experiment_name))
    params = Params(constants.params_path.format(dataset_name, experiment_name))
    stats = Params(constants.stats_path.format(dataset_name, experiment_name))
    validate_params(params)
    prints.show_training_info(params)

    training_dataloader, validation_dataloader = training_setup.prepare_dataloaders(params, config)
    prints.print_dataset_stats(training_dataloader, validation_dataloader)

    model = training_setup.model_setup(params, config)
    optimizer = training_setup.optimizer_setup(model, params)
    start_epoch = 0
    if hasattr(params, "load_type"):
        start_epoch = training_setup.load_model(
            model, optimizer, params, dataset_name, experiment_name)
    else:
        if amp_available and general_config.use_amp:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=general_config.amp_opt_level
            )
    prints.print_trained_parameters_count(model, optimizer)

    experiment_info = prints.create_tensorboard_name(dataset_name, experiment_name, params)

    if config.model_id in constants.segmentor_ids:
        model_trainer = train.Segmentation_Trainer(model=model,
                                                   training_dataloader=training_dataloader,
                                                   validation_dataloader=validation_dataloader,
                                                   optimizer=optimizer, params=params,
                                                   config=config, stats=stats,
                                                   start_epoch=start_epoch,
                                                   dataset_name=dataset_name,
                                                   experiment_info=experiment_info,
                                                   experiment_name=experiment_name)
    elif config.model_id in constants.detectors:
        model_trainer = train.Detector_Trainer(model=model, training_dataloader=training_dataloader,
                                               validation_dataloader=validation_dataloader,
                                               optimizer=optimizer, params=params, config=config,
                                               stats=stats,
                                               start_epoch=start_epoch,
                                               dataset_name=dataset_name,
                                               experiment_info=experiment_info,
                                               experiment_name=experiment_name)
    if train_model:
        model_trainer.train()

    if evaluate_model:
        model_trainer.evaluate(start_epoch, no_saving=True)

    if compute_dset_mean_std:
        print("Computing dataset mean and std!")
        images = training_dataloader.dataset.get_images()
        mean, std = data_normalization.per_dataset_norm(images)
        print("Dataset mean and standard deviation: ", mean, std)

    if compute_dset_gt_bounds:
        roi_crop.get_dataset_gt_bounds(dataset_name, params, config)


if __name__ == '__main__':
    main()

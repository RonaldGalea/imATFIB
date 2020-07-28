import numpy as np
import argparse
import torch
import cv2
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
from utils.ROI_crop import roi_crop, predict_ROI
from utils import visualization
from utils import prepare_models


def main(experiment_name=None):
    parser = argparse.ArgumentParser(description='Run Settings.')
    parser.add_argument('--dataset_name', dest="dataset_name",
                        help='mmwhs/imatfib-whs/ACDC_training', default=constants.mmwhs_root_dir)
    parser.add_argument('--experiment_name', dest="experiment_name",
                        help='experiment root folder', default=constants.unet)
    parser.add_argument('--train_model', dest="train_model", type=bool,
                        help='trains model',
                        default=False)
    parser.add_argument('--evaluate_model', dest="evaluate_model", type=bool,
                        help='evaluates model, remember to load model first!',
                        default=True)
    parser.add_argument('--view_results', dest="view_results", nargs='+',
                        help='ids of models whose results are to be visualized')
    parser.add_argument('--show_results', dest="show_results",
                        help='if True, results are displayed, otherwise they are saved to a folder')
    parser.add_argument('--show_detection_results', dest="show_detection_results",
                        help='if True, detection results are displayed')
    parser.add_argument('--compute_dset_mean_std', dest="compute_dset_mean_std", type=bool,
                        help='computes the mean and std of the dataset, should then set corresponding values in general_config',
                        default=False)
    parser.add_argument('--compute_dset_gt_bounds', dest="compute_dset_gt_bounds", type=bool,
                        help='computes bounds of the labeled area of the dataset',
                        default=False)

    args = parser.parse_args([])
    if experiment_name:
        args.experiment_name = experiment_name
    print("Args in main: ", args, "\n")
    validate_args(args)

    if args.view_results:
        model_ids = args.view_results
        print("Showing results on ", len(model_ids), " models")
        models, params_list, configs_list = prepare_models.prepare(model_ids,
                                                                   args.dataset_name)

        validation_loader = training_setup.prepare_val_loader(args.dataset_name, params_list[0],
                                                              configs_list[0])

        visualization.visualize_validation_dataset(validation_loader, models, params_list[0],
                                                   configs_list[0], model_ids, args.show_results)
    if args.show_detection_results:
        model_ids = args.view_results
        models, params_list, configs_list = prepare_models.prepare(model_ids,
                                                                   args.dataset_name)
        seg_params, detect_params = params_list
        seg_config, detect_config = configs_list

        # when using another model to get the ROI for the segmentor, the validation loader should
        # be a segmentor type, but with the height and width desired by the detector
        seg_params.default_width, seg_params.default_height = detect_params.default_width, detect_params.default_height
        validation_loader = training_setup.prepare_val_loader(args.dataset_name, seg_params,
                                                              seg_config)

        predict_ROI.validate_using_predicted_ROI(validation_loader, models[0], models[1],
                                                 seg_params, detect_params, seg_config,
                                                 detect_config)

    if not (args.train_model or args.evaluate_model):
        return

    config = Params(constants.config_path.format(args.dataset_name, args.experiment_name))
    params = Params(constants.params_path.format(args.dataset_name, args.experiment_name))
    stats = Params(constants.stats_path.format(args.dataset_name, args.experiment_name))
    validate_params(params)
    prints.show_training_info(params)

    training_dataloader, validation_dataloader = training_setup.prepare_dataloaders(args.dataset_name,
                                                                                    params,
                                                                                    config)
    prints.print_dataset_stats(training_dataloader, validation_dataloader)

    model = training_setup.model_setup(args.dataset_name, params, config)
    optimizer = training_setup.optimizer_setup(model, params)
    start_epoch = 0
    if hasattr(params, "load_type"):
        start_epoch = training_setup.load_model(
            model, optimizer, params, args.dataset_name, args.experiment_name)
    else:
        if amp_available and general_config.use_amp:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=general_config.amp_opt_level
            )
    prints.print_trained_parameters_count(model, optimizer)

    experiment_info = prints.create_tensorboard_name(args, params)

    if config.model_id in constants.segmentor_ids:
        model_trainer = train.Segmentation_Trainer(model=model,
                                                   training_dataloader=training_dataloader,
                                                   validation_dataloader=validation_dataloader,
                                                   optimizer=optimizer, params=params,
                                                   config=config, stats=stats,
                                                   start_epoch=start_epoch,
                                                   dataset_name=args.dataset_name,
                                                   experiment_info=experiment_info,
                                                   experiment_name=args.experiment_name)
    elif config.model_id in constants.detectors:
        model_trainer = train.Detector_Trainer(model=model, training_dataloader=training_dataloader,
                                               validation_dataloader=validation_dataloader,
                                               optimizer=optimizer, params=params, config=config,
                                               stats=stats,
                                               start_epoch=start_epoch,
                                               dataset_name=args.dataset_name,
                                               experiment_info=experiment_info,
                                               experiment_name=args.experiment_name)
    if args.train_model:
        model_trainer.train()

    if args.evaluate_model:
        model_trainer.evaluate(start_epoch)

    if args.compute_dset_mean_std:
        print("Computing dataset mean and std!")
        images = training_dataloader.dataset.get_images()
        mean, std = data_normalization.per_dataset_norm(images)
        print("Dataset mean and standard deviation: ", mean, std)

    if args.compute_dset_gt_bounds:
        roi_crop.get_dataset_gt_bounds(args.dataset_name, params, config)


def validate_args(args):
    valid_dsets = [constants.acdc_root_dir, constants.imatfib_root_dir, constants.mmwhs_root_dir]
    if args.dataset_name not in valid_dsets:
        raise AssertionError('Invalid dataset name')
    if args.view_results:
        for name in args.view_results[:-1]:
            if name not in constants.segmentor_ids:
                raise AssertionError("Invalid model id")


if __name__ == '__main__':
    main()

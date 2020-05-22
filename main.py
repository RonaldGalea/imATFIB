import numpy as np
import argparse
import torch
import cv2
from torch.utils.tensorboard import SummaryWriter
try:
    from apex import amp
    amp_available = True
except ImportError:
    print("Cannot import NVIDIA Apex...")
from random import randint

import general_config
import constants
from training import train
from utils.dataset_utils import data_normalization
from utils.params import Params
from utils.training_utils import prints, training_processing, training_setup
from utils.ROI_crop import roi_crop
from utils import visualization


def main():
    parser = argparse.ArgumentParser(description='Run Settings.')
    parser.add_argument('-dataset_name', dest="dataset_name",
                        help='mmwhs/imatfib-whs/ACDC_training', default=constants.imatfib_root_dir)
    parser.add_argument('-experiment_name', dest="experiment_name",
                        help='experiment root folder', default=constants.deeplab)
    parser.add_argument('-load_model', dest="load_model", type=bool,
                        help='lodel model weights and optimizer at specified experiment',
                        default=False)
    parser.add_argument('-train_model', dest="train_model", type=bool,
                        help='trains model, from checkpoint if load_model else from scratch',
                        default=True)
    parser.add_argument('-evaluate_model', dest="evaluate_model", type=bool,
                        help='evaluates model, load_model should be true when this is true',
                        default=False)
    parser.add_argument('-view_results', dest="view_results", type=bool,
                        help='visualize model results on the validation set',
                        default=False)
    parser.add_argument('-inspect_train_results', dest="inspect_train_results", type=bool,
                        help='visualize model results on the training set, augmentations inlcuded',
                        default=False)
    parser.add_argument('-compute_dset_mean_std', dest="compute_dset_mean_std", type=bool,
                        help='computes the mean and std of the dataset, should then set corresponding values in general_config',
                        default=False)
    parser.add_argument('-compute_dset_gt_bounds', dest="compute_dset_gt_bounds", type=bool,
                        help='computes bounds of the labeled area of the dataset',
                        default=False)

    args = parser.parse_args(args=[])
    print("Args in main: ", args, "\n")
    validate_args(args)

    params = Params(constants.params_path.format(args.dataset_name, args.experiment_name))
    stats = Params(constants.stats_path.format(args.dataset_name, args.experiment_name))
    validate_params(params)
    prints.show_training_info(params)

    training_dataloader, validation_dataloader = training_setup.prepare_dataloaders(args.dataset_name,
                                                                                    params)
    prints.print_dataset_stats(training_dataloader, validation_dataloader)

    model = training_setup.model_setup(args.dataset_name, params)
    optimizer = training_setup.optimizer_setup(model, params)
    start_epoch = 0
    if args.load_model:
        start_epoch = training_setup.load_model(model, optimizer,
                                                params, args.dataset_name)
    else:
        if amp_available and general_config.use_amp:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=general_config.amp_opt_level
            )
    prints.print_trained_parameters_count(model, optimizer)

    rand_id = randint(0, 10000)
    experiment_info = args.experiment_name + "/" + args.dataset_name + "_" + str(params.n_epochs) + "_" + str(params.data_augmentation) + "_" + str(params.roi_crop) + "+" + str(params.default_width) + "_" + str(rand_id)
    writer = SummaryWriter(log_dir="runs/"+experiment_info, filename_suffix=params.model_id)
    # !!!!!!!move writer definition in self.train of model trainer to stop spam


    model_trainer = train.Model_Trainer(model=model, training_dataloader=training_dataloader,
                                        validation_dataloader=validation_dataloader,
                                        optimizer=optimizer, params=params, stats=stats,
                                        start_epoch=start_epoch, dataset_name=args.dataset_name,
                                        writer=writer)
    if args.train_model:
        model_trainer.train()

    if args.evaluate_model:
        model_trainer.evaluate(start_epoch)

    if args.view_results:
        model.eval()
        with torch.no_grad():
            for volume, mask, r_info in validation_dataloader:
                print("In main: ", volume.shape, mask.shape, torch.mean(volume), torch.std(volume))
                volume, mask = volume.to(general_config.device), mask.to(general_config.device)
                processed_volume = training_processing.process_volume(model, volume, mask, r_info)
                dice, concrete_volume, mask = training_processing.compute_dice(processed_volume,
                                                                               mask)

                volume = volume.cpu().numpy()
                for img_slice, pred_slice, mask_slice in zip(volume, concrete_volume, mask):
                    print("Input shape: ", img_slice.shape)
                    visualization.show_image2d(img_slice, "input", unnorm=True)
                    print("Pred shape: ", pred_slice.shape)
                    visualization.show_image2d(pred_slice, "pred")
                    print("Label shape: ", mask_slice.shape)
                    visualization.show_image2d(mask_slice, "label")
                    cv2.waitKey(0)
                cv2.destroyAllWindows()
                print(dice)

    if args.inspect_train_results:
        model.eval()
        with torch.no_grad():
            for image, mask in training_dataloader:
                print("In main: ", image.shape)
                image, mask = image.to(general_config.device), mask.to(general_config.device)
                prediction = model(image)
                dice, prediction, mask = training_processing.compute_dice(prediction,
                                                                          mask)
                image = image.cpu().numpy()
                for img_slice, pred_slice, mask_slice in zip(image, prediction, mask):
                    img_slice = img_slice.squeeze(0)
                    print("Input shape: ", img_slice.shape)
                    visualization.show_image2d(img_slice, "input")
                    print("Pred shape: ", pred_slice.shape)
                    visualization.show_image2d(pred_slice, "pred")
                    print("Label shape: ", mask_slice.shape)
                    visualization.show_image2d(mask_slice, "label")
                    cv2.waitKey(0)
                cv2.destroyAllWindows()

                print(dice)

    if args.compute_dset_mean_std:
        images = training_dataloader.dataset.get_images()
        mean, std = data_normalization.per_dataset_norm(images)
        print("Dataset mean and standard deviation: ", mean, std)

    if args.compute_dset_gt_bounds:
        roi_crop.get_dataset_gt_bounds(args.dataset_name, params)


def validate_args(args):
    valid_dsets = [constants.acdc_root_dir, constants.imatfib_root_dir, constants.mmwhs_root_dir]
    if args.dataset_name not in valid_dsets:
        raise AssertionError('Invalid dataset name')


def validate_params(params):
    norm_type = params.norm_type
    data_augmentation = params.data_augmentation
    lr_decay = params.lr_decay
    roi_crop = params.roi_crop
    if norm_type not in constants.norm_types:
        raise AssertionError("Params not ok..." + "norm_type")
    if data_augmentation not in constants.aug_types:
        raise AssertionError("Params not ok..." + "data_augmentation")
    if lr_decay not in constants.lr_schedulers:
        raise AssertionError("Params not ok..." + "lr_decay")
    if roi_crop not in constants.roi_types:
        raise AssertionError("Params not ok..." + "roi_crop")


if __name__ == '__main__':
    main()

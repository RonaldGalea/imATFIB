import numpy as np
import argparse
import torch
try:
    from apex import amp
    amp_available = True
except ImportError:
    print("Cannot import NVIDIA Apex...")


import general_config
import constants
from training import train
from utils.params import Params
from utils.training_utils import prints, training_processing, training_setup
from utils import visualization


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-dataset_name', dest="dataset_name",
                        help='mmwhs/imatfib-whs/ACDC_training', default=constants.imatfib_root_dir)
    parser.add_argument('-experiment_name', dest="experiment_name",
                        help='experiment root folder', default=constants.unet)
    parser.add_argument('-load_model', dest="load_model", type=bool,
                        help='lodel model weights and optimizer at specified experiment',
                        default=True)
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

    model_trainer = train.Model_Trainer(model=model, training_dataloader=training_dataloader,
                                        validation_dataloader=validation_dataloader,
                                        optimizer=optimizer, params=params, stats=stats,
                                        start_epoch=start_epoch, dataset_name=args.dataset_name)
    if args.train_model:
        model_trainer.train()

    if args.evaluate_model:
        model_trainer.evaluate()

    if args.view_results:
        model.eval()
        with torch.no_grad():
            for volume, mask, info in validation_dataloader:
                print("In main: ", volume.shape)
                volume, mask = volume.to(general_config.device), mask.to(general_config.device)
                processed_volume = training_processing.process_volume(model, volume, mask)
                dice, concrete_volume, mask = training_processing.compute_dice(processed_volume,
                                                                               mask)
                visualization.visualize_img_mask_pair(np.transpose(concrete_volume, (1, 2, 0)),
                                                      np.transpose(mask, (1, 2, 0)))
                print(dice)


def validate_args(args):
    valid_dsets = [constants.acdc_root_dir, constants.imatfib_root_dir, constants.mmwhs_root_dir]
    if args.dataset_name not in valid_dsets:
        raise AssertionError('Invalid dataset name')


def validate_params(params):
    valid_segs = [constants.whole_heart_seg, constants.multi_class_seg]
    if params.seg_type not in valid_segs:
        raise AssertionError('Invalid Segmentation type')


if __name__ == '__main__':
    main()

from pathlib import Path
import argparse
try:
    from apex import amp
except ImportError:
    print("Cannot import NVIDIA Apex...")


import general_config
import constants
from training import train
from utils.params import Params
from utils import training_setup


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-dataset_name', dest="dataset_name",
                        help='mmwhs/imatfib-whs/ACDC_training', default=constants.imatfib_root_dir)
    parser.add_argument('-experiment_name', dest="experiment_name",
                        help='experiment root folder', default=constants.unet)
    parser.add_argument('-load_model', dest="load_model",
                        help='lodel model weights and optimizer at specified experiment',
                        default=constants.unet)

    args = parser.parse_args()
    print("Args in main: ", args, "\n")
    validate_args(args)

    params = Params(constants.params_path.format(args.dataset_name, args.experiment_name))
    validate_params(params)

    training_dataloader, validation_dataloader = training_setup.prepare_dataloaders(args.dataset_name,
                                                                                    params)
    model = training_setup.model_setup(args.dset_name, params)

    model_trainer = train.Model_Trainer(model="dummy", training_dataloader=training_dataloader,
                                        validation_dataloader=validation_dataloader, params=params)
    model_trainer.train()


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

from pathlib import Path
import argparse
import general_config
import constants
from data_loading import create_datasets


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-dataset_name', dest="dataset_name",
                        help='mmwhs/imatfib/acdc', default=constants.acdc_root_dir)
    parser.add_argument('-seg_type', dest="seg_type",
                        help='multiple-classes/whole-heart', default=constants.multi_class_seg)
    parser.add_argument('-k_split', dest="k_split", type=int,
                        help='every kth element goes in the validation set (a k_split of 2 splits the dataset equally between train and val)',
                        default='5')
    parser.add_argument('-split_train_val', dest="split_train_val", type=bool,
                        help='if False, train on whole dataset', default=True)

    args = parser.parse_args()
    print("Args in main: ", args, "\n")
    validate_args(args)

    training_dataset, validation_dataset = create_datasets.train_val(dataset_name=args.dataset_name,
                                                                     k_split=args.k_split,
                                                                     seg_type=args.seg_type,
                                                                     split_train_val=args.split_train_val)

    for image in training_dataset:
        print("In main: ", image, "\n")

    for image in validation_dataset:
        print("In main: ", image, "\n")


def validate_args(args):
    if args.dataset_name == constants.acdc_root_dir and args.seg_type == constants.whole_heart_seg:
        raise AssertionError('ACDC does not have whole-heart by default')

    valid_dsets = [constants.acdc_root_dir, constants.imatfib_root_dir, constants.mmwhs_root_dir]
    if args.dataset_name not in valid_dsets:
        raise AssertionError('Invalid dataset name')

    valid_segs = [constants.whole_heart_seg, constants.multi_class_seg]
    if args.seg_type not in valid_segs:
        raise AssertionError('Invalid Segmentation type')


if __name__ == '__main__':
    main()

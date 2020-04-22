from pathlib import Path
from utils import acdc, general
from data_loading import dataset
import general_config
import constants


def train_val(dataset_name, k_split, seg_type, split_train_val):
    """
    Args:
    dataset_name: string - root training dataset directory
    k_split: int - every kth element goes in the validation set
    (a k_split of 2 splits the dataset equally between train and val)
    seg_type: string - multiple-classes or whole-heart
    split_train_val: bool - if False, train on whole dataset

    This function splits the samples from the dataset directory in two sets: train and val,
    creating two Dataset objects using them

    For the ACDC dataset the split is fixed, exactly as done by Baumgarter et al.
    For imogen and mmwhs, the split factor is controlled by k_split

    Return: the training and validation Dataset objects (torch.utils.data.Dataset)
    """
    if general_config.read_numpy:
        dataset_name += '_npy'

    dataset_dir = Path.cwd() / 'datasets' / dataset_name
    if constants.acdc_root_dir in dataset_name:
        split_dict = acdc.acdc_train_val_split(dataset_dir)
    else:
        split_dict = general.train_val_split(dataset_dir, k_split=k_split,
                                             split_train_val=split_train_val)

    print("TRAIN")
    for file in split_dict['train']:
        print(str(file) + "\n")

    print("\n\n")

    print("VALIDATION")
    for file in split_dict['val']:
        print(str(file) + "\n")

    training_dataset = dataset.MRI_Dataset(dset_name=dataset_name, dset_type='train',
                                           paths=split_dict['train'], seg_type=seg_type)
    validation_dataset = dataset.MRI_Dataset(dset_name=dataset_name, dset_type='val',
                                             paths=split_dict['val'], seg_type=seg_type)

    return training_dataset, validation_dataset

from pathlib import Path
from utils import acdc, general
from data_loading import dataset_2d, dataset_3d
import general_config
import constants


def train_val(dataset_name, params):
    """
    Args:
    dataset_name: string - root training dataset directory
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
        split_dict = general.train_val_split(dataset_dir, k_split=params.k_split,
                                             split_train_val=params.split_train_val)

    training_dataset = dataset_2d.MRI_Dataset_2d(dset_name=dataset_name, dset_type='train',
                                           paths=split_dict['train'], params=params)
    validation_dataset = dataset_3d.MRI_Dataset_3d(dset_name=dataset_name, dset_type='val',
                                             paths=split_dict['val'], params=params)

    return training_dataset, validation_dataset

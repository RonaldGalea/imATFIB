from utils.dataset_utils import reading
from data_loading import dataset_2d, dataset_3d


def train_val(dataset_name, params):
    """
    Args:
    dataset_name: string - root training dataset directory

    Return: the training and validation Dataset objects (torch.utils.data.Dataset)
    """
    split_dict = reading.get_train_val_paths(dataset_name, params.k_split)

    training_dataset = dataset_2d.MRI_Dataset_2d(dset_name=dataset_name, dset_type='train',
                                                 paths=split_dict['train'], params=params)
    validation_dataset = dataset_3d.MRI_Dataset_3d(dset_name=dataset_name, dset_type='val',
                                                   paths=split_dict['val'], params=params)

    return training_dataset, validation_dataset

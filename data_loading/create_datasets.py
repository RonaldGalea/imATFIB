import constants
from utils.dataset_utils import reading
from utils.dataset_structuring import acdc
from data_loading import dataset_2d, dataset_3d


def train_val(dataset_name, params, config):
    """
    Args:
    dataset_name: string - root training dataset directory

    Return: the training and validation Dataset objects (torch.utils.data.Dataset)
    """
    split_dict = reading.get_train_val_paths(dataset_name, params.k_split)

    if config.model_id in constants.segmentor_ids:
        training_dataset = dataset_2d.MRI_Dataset_2d_Segmentation(dset_name=dataset_name,
                                                                  dset_type='train',
                                                                  paths=split_dict['train'],
                                                                  params=params, config=config)
        validation_dataset = dataset_3d.MRI_Dataset_3d_Segmentation(dset_name=dataset_name,
                                                                    dset_type='val',
                                                                    paths=split_dict['val'],
                                                                    params=params, config=config)
    else:
        training_dataset = dataset_2d.MRI_Dataset_2d_Detection(dset_name=dataset_name,
                                                               dset_type='train',
                                                               paths=split_dict['train'],
                                                               params=params, config=config)
        validation_dataset = dataset_3d.MRI_Dataset_3d_Detection(dset_name=dataset_name,
                                                                 dset_type='val',
                                                                 paths=split_dict['val'],
                                                                 params=params, config=config)

    return training_dataset, validation_dataset


def create_val_set(dataset_name, params, config):
    split_dict = reading.get_train_val_paths(dataset_name, params.k_split)

    if config.model_id in constants.segmentor_ids:
        validation_dataset = dataset_3d.MRI_Dataset_3d_Segmentation(dset_name=dataset_name,
                                                                    dset_type='val',
                                                                    paths=split_dict['val'],
                                                                    params=params,
                                                                    config=config)
    else:
        validation_dataset = dataset_3d.MRI_Dataset_3d_Detection(dset_name=dataset_name,
                                                                 dset_type='val',
                                                                 paths=split_dict['val'],
                                                                 params=params,
                                                                 config=config)
    return validation_dataset


def create_test_set(dataset_name, params, config):
    split_dict = reading.get_train_val_paths(dataset_name, params.k_split)
    test_list = split_dict['train'] + split_dict['val']
    # test_list = split_dict['val']

    test_dataset = dataset_3d.MRI_Dataset_3d_Segmentation_Test(dset_name=dataset_name,
                                                               dset_type='val',
                                                               paths=test_list,
                                                               params=params,
                                                               config=config)

    return test_dataset

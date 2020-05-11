from torch.utils.data import Dataset

import general_config
from utils import visualization
from data_loading import data_augmentation


class MRI_Dataset(Dataset):
    """
    Base Dataset class, MRI_Dataset_2d and MRI_Dataset_3d inherit from this
    """

    def __init__(self, dset_name, dset_type, paths, params):
        """
        Args:
        dset_name - string: name of the dataset
        dset_type - string: train or val
        paths - list of Path: paths to data samples
        params - params.json of current experiment
        """
        self.dset_name = dset_name
        self.dset_type = dset_type
        self.paths = paths
        self.seg_type = general_config.seg_type
        self.norm_type = params.norm_type
        self.params = params
        self.augmentor = data_augmentation.Augmentor(params)
        self.load_everything_in_memory()
        if general_config.visualize_dataset:
            self.visualize_dataset_samples()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, batched_indices):
        raise NotImplementedError

    def load_everything_in_memory(self):
        raise NotImplementedError

    def visualize_dataset_samples(self):
        for image, mask in zip(self.images, self.masks):
            if len(image.shape) == 3:
                visualization.visualize_img_mask_pair(image, mask)
            else:
                visualization.visualize_img_mask_pair_2d(image, mask)
            exit = input("exit? y/n")
            if exit == 'y':
                return

    def dataset_info(self):
        print("Dataset name: ", self.dset_name, "\n")
        print("Dataset type: ", self.dset_type, "\n")
        print("Elements in dataset: ", len(self.paths), "\n")
        print("Segmentation type: ", self.seg_type, "\n")

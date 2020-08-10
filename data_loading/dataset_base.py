import cv2
import torch

from torch.utils.data import Dataset

import general_config
import constants
import general_dataset_settings
from utils import visualization
from data_loading import data_augmentation
from utils.training_utils import box_utils


class MRI_Dataset(Dataset):
    """
    Base Dataset class, MRI_Dataset_2d and MRI_Dataset_3d inherit from this
    """

    def __init__(self, dset_name, dset_type, paths, params, config):
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
        self.seg_type = config.seg_type
        self.norm_type = params.norm_type
        self.params = params
        self.config = config
        self.augmentor = data_augmentation.Augmentor(params, config)
        self.load_everything_in_memory()
        self.initialize_elements()
        if config.visualize_dataset:
            self.visualize_dataset_samples()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, batched_indices):
        raise NotImplementedError

    def load_everything_in_memory(self):
        raise NotImplementedError

    def visualize_dataset_samples(self):
        for path, image, mask in zip(self.paths, self.images, self.masks):
            print("image path", path)
            if len(image.shape) == 3:
                visualization.visualize_img_mask_pair(image, mask)
            else:
                visualization.visualize_img_mask_pair_2d(image, mask)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def dataset_info(self):
        print("Dataset name: ", self.dset_name, "\n")
        print("Dataset type: ", self.dset_type, "\n")
        print("Elements in dataset: ", len(self.paths), "\n")
        print("Segmentation type: ", self.seg_type, "\n")

    def initialize_elements(self):
        if self.config.dataset == constants.imatfib_root_dir:
            self.dataset_mean = general_dataset_settings.imatfib_dataset_mean
            self.dataset_std = general_dataset_settings.imatfib_dataset_std

        elif self.config.dataset == constants.acdc_root_dir or self.config.dataset == constants.acdc_test_dir:
            self.dataset_mean = general_dataset_settings.acdc_dataset_mean
            self.dataset_std = general_dataset_settings.acdc_dataset_std

        else:
            self.dataset_mean = general_dataset_settings.mmwhs_dataset_mean
            self.dataset_std = general_dataset_settings.mmwhs_dataset_std

        if self.params.norm_type == constants.per_slice:
            print("Using per slice normalization!")
        else:
            print("Dataset mean and std: ", self.dataset_mean, self.dataset_std)

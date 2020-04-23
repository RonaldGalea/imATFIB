import torch
import numpy as np
from torch.utils.data import Dataset

import general_config
from utils import reading, visualization, statistics
from data_loading import data_augmentation


class MRI_Dataset(Dataset):
    """
    Class to handle data loading
    Receives a list of paths to the input samples, and finds the corresponding ground truth masks
    depending on the current directory structure
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
        self.seg_type = params.seg_type
        self.norm_type = params.norm_type
        self.augmentor = data_augmentation.Augmentor(params)
        self.load_everything_in_memory()
        self.necessary_preprocessing()
        self.visualize_dataset_samples()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image, mask = self.images[idx], self.masks[idx]
        print("In dataset, image shape: ", image.shape)

        return self.paths[idx]

    def load_everything_in_memory(self):
        """
        Since the datasets are small enough to be loaded wholely in memory....

        self.images = list of volumes
        self.masks = labels for those volumes
        self.info = list of [affine, header] for each volume
        """
        images, masks, infos = [], [], []
        for path in self.paths:
            image, mask, info = reading.get_img_mask_pair(image_path=path,
                                                          numpy=general_config.read_numpy,
                                                          dset_name=self.dset_name,
                                                          seg_type=self.seg_type)
            # permute dimensions to D x H x W for easier indexing
            # after processing will be switched back
            images.append(np.transpose(image, (2, 0, 1)))
            masks.append(np.transpose(mask, (2, 0, 1)))
            infos.append(info)

        self.images = images
        self.masks = masks
        self.info = info

    def necessary_preprocessing(self):
        for i, (image, mask) in enumerate(zip(self.images, self.masks)):
            resized_image, resized_mask = self.augmentor.resize_volume_HW(image, mask)
            self.images[i] = resized_image
            self.masks[i] = resized_mask

        statistics.normalize(self.images, self.norm_type)

    def visualize_dataset_samples(self):
        for image, mask in zip(self.images, self.masks):
            visualization.visualize_img_mask_pair(np.transpose(image, (1, 2, 0)),
                                                  np.transpose(mask, (1, 2, 0)))

    def dataset_info(self):
        print("Dataset name: ", self.dset_name, "\n")
        print("Dataset type: ", self.dset_type, "\n")
        print("Elements in dataset: ", len(self.paths), "\n")
        print("Segmentation type: ", self.seg_type, "\n")

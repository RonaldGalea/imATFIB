import torch
import numpy as np
from torch.utils.data import Dataset

import general_config
from utils import reading, visualization


class MRI_Dataset(Dataset):
    """
    Class to handle data loading
    Receives a list of paths to the input samples, and finds the corresponding ground truth masks
    depending on the current directory structure
    """

    def __init__(self, dset_name, dset_type, paths, seg_type):
        """
        Args:
        dset_name - string: name of the dataset
        dset_type - string: train or val
        paths - list of Path: paths to data samples
        """
        self.dset_name = dset_name
        self.dset_type = dset_type
        self.paths = paths
        self.seg_type = seg_type
        self.load_everything_in_memory()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image, mask = self.images[idx], self.masks[idx]

        print("In dataset, image shape: ", image.shape)
        visualization.visualize_img_mask_pair(image, mask)

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
            images.append(image)
            masks.append(mask)
            infos.append(info)
        self.images = images
        self.masks = masks
        self.info = info

    def necessary_preprocessing(self):
        pass

    def augmentation(self):
        pass

    def dataset_info(self):
        print("Dataset name: ", self.dset_name, "\n")
        print("Dataset type: ", self.dset_type, "\n")
        print("Elements in dataset: ", len(self.paths), "\n")
        print("Segmentation type: ", self.seg_type, "\n")

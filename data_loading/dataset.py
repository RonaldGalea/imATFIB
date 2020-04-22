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
        paths - list of Path: paths to data samples (representing data ids)
        """
        self.dset_name = dset_name
        self.dset_type = dset_type
        self.ids = paths
        self.seg_type = seg_type

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        current_id = self.ids[idx]
        image, mask = reading.get_img_mask_pair(image_path=current_id,
                                                numpy=general_config.read_numpy,
                                                dset_name=self.dset_name, seg_type=self.seg_type)
        visualization.visualize_img_mask_pair(image, mask)

        return current_id

    def get_ids(self):
        self.ids = []
        for image_path in self.images_path.glob('*'):
            self.ids.append(image_path.stem + image_path.suffix)

        print("Number of items in dataset: ", len(self.ids), "\n")

    def dataset_info(self):
        print("Dataset name: ", self.dset_name, "\n")
        print("Dataset type: ", self.dset_type, "\n")
        print("Elements in dataset: ", len(self.ids), "\n")
        print("Segmentation type: ", self.seg_type, "\n")

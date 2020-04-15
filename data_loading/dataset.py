import torch
import numpy as np
from torch.utils.data import Dataset

from utils import reading, visualization


class MRI_Dataset(Dataset):
    """
    Class to handle data loading
    """
    def __init__(self, images_path, masks_path):
        """
        Args:
        images_path - pathlib.Path: path to images folder
        masks_path - pathlib.Path: path to mask folder

        """
        self.images_path = images_path
        self.masks_path = masks_path
        self.get_ids()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        current_id = self.ids[idx]
        current_img_path = self.images_path / current_id
        image, mask = reading.get_img_mask_pair(current_img_path, self.masks_path)
        visualization.visualize_img_mask_pair(image, mask)

        return current_id

    def get_ids(self):
        self.ids = []
        for image_path in self.images_path.glob('**/*'):
            self.ids.append(image_path.stem + image_path.suffix)

        print("Number of items in dataset: ", len(self.ids), "\n")

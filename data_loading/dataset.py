import torch
import numpy as np
from torch.utils.data import Dataset

from utils import reading, visualization


class MRI_Dataset(Dataset):

    def __init__(self, images_path, gts_path):
        """
        Args:

        """
        self.images_path = images_path
        self.gts_path = gts_path
        self.get_ids()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        current_id = self.ids[idx]
        current_img_path = self.images_path / current_id
        image, gt = reading.get_img_gt_pair(current_img_path, self.gts_path)
        visualization.visualize_img_gt_pair(image, gt)

        return current_id

    def get_ids(self):
        self.ids = []
        for image_path in self.images_path.glob('**/*'):
            self.ids.append(image_path.stem + image_path.suffix)

        print("Number of items in dataset: ", len(self.ids), "\n")

import torch
import numpy as np

import general_config
import constants
from data_loading import dataset_base
from utils.dataset_utils import reading


class MRI_Dataset_3d(dataset_base.MRI_Dataset):
    """
    Class to handle data for the validation set

    Returns a volume at a time, with untouched masks, for correct evaluation
    """
    def __init__(self, dset_name, dset_type, paths, params):
        super(MRI_Dataset_3d, self).__init__(dset_name, dset_type, paths, params)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, batched_indices):
        """
        Returns:
        torch.tensor: D x H x W (float32 image)
        torch.tensor: D x H x W (int64 mask)
        """
        idx = batched_indices[0]
        image, mask = self.images[idx], self.masks[idx]
        # D x H x W
        image, mask = np.transpose(image, (2, 0, 1)), np.transpose(mask, (2, 0, 1))
        image, _ = self.augmentor.resize_volume_HW(image, mask)

        reconstruction_info = None
        if self.params.roi_crop != constants.no_roi_extraction:
            image, reconstruction_info = self.augmentor.extract_ROI_3d(image, mask)

        # torch tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        normalized_image = []
        for slice in image:
            heigth, width = slice.shape
            slice = slice.view(1, heigth, width)
            slice = self.augmentor.normalize(slice)
            slice = slice.view(heigth, width)
            normalized_image.append(slice)
        image = torch.stack(normalized_image)

        mask = mask.to(torch.int64)

        return image, mask, reconstruction_info

    def load_everything_in_memory(self):
        """
        Since the datasets are small enough to be loaded wholely in memory....

        self.images = list of 3D volumes
        self.masks = labels for those volumes
        """
        images, masks = [], []
        for path in self.paths:
            image, mask, info = reading.get_img_mask_pair(image_path=path,
                                                          numpy=general_config.read_numpy,
                                                          dset_name=self.dset_name,
                                                          seg_type=self.seg_type)
            images.append(image)
            masks.append(mask)

        self.images = images
        self.masks = masks

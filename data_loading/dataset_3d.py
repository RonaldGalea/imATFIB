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
    def __init__(self, dset_name, dset_type, paths, params, config):
        super(MRI_Dataset_3d, self).__init__(dset_name, dset_type, paths, params, config)
        self.visualization_mode = False

    def __len__(self):
        return len(self.paths)

    def load_everything_in_memory(self):
        """
        Since the datasets are small enough to be loaded wholely in memory....

        self.images = list of 3D volumes
        self.masks = labels for those volumes
        """
        self.images, self.masks, self.infos = [], [], []
        for path in self.paths:
            image, mask, info = reading.get_img_mask_pair(image_path=path,
                                                          dset_name=self.dset_name,
                                                          seg_type=self.seg_type)
            self.images.append(image)
            self.masks.append(mask)
            self.infos.append(info)


class MRI_Dataset_3d_Segmentation(MRI_Dataset_3d):
    def __init__(self, dset_name, dset_type, paths, params, config):
        super(MRI_Dataset_3d_Segmentation, self).__init__(
            dset_name, dset_type, paths, params, config)

    def __getitem__(self, batched_indices):
        """
        Returns:
        torch.tensor: D x H x W (float32 image)
        torch.tensor: D x H x W (int64 mask)
        """
        idx = batched_indices[0]
        image, mask, header_info = self.images[idx], self.masks[idx], self.infos[idx]
        # D x H x W
        image, mask = np.transpose(image, (2, 0, 1)), np.transpose(mask, (2, 0, 1))
        image, reconstruction_info = self.augmentor.segmentor_valid_data(image, mask)

        # torch tensors
        mask = mask.astype(np.int64)
        mask = torch.from_numpy(mask)

        image = torch.from_numpy(image)
        normalized_image = []
        for slice in image:
            heigth, width = slice.shape
            slice = slice.view(1, heigth, width)
            slice = self.augmentor.normalize(slice)
            slice = slice.view(heigth, width)
            normalized_image.append(slice)
        image = torch.stack(normalized_image)

        if self.visualization_mode:
            return image, mask, reconstruction_info, self.images[idx]
        return image, mask, reconstruction_info, header_info


class MRI_Dataset_3d_Detection(MRI_Dataset_3d):
    def __init__(self, dset_name, dset_type, paths, params, config):
        super(MRI_Dataset_3d_Detection, self).__init__(dset_name, dset_type, paths, params, config)

    def __getitem__(self, batched_indices):
        """
        Returns:
        torch.tensor: D x H x W (float32 image)
        torch.tensor: D x 5
        """
        coords_n_scores = None
        idx = batched_indices[0]
        image, mask = self.images[idx], self.masks[idx]
        # D x H x W
        image, mask = np.transpose(image, (2, 0, 1)), np.transpose(mask, (2, 0, 1))

        image, coords_n_scores = self.augmentor.detector_valid_data(image, mask)
        coords_n_scores = torch.tensor(coords_n_scores)
        coords_n_scores = coords_n_scores.to(torch.float32)

        image = torch.from_numpy(image)
        normalized_image = []
        for slice in image:
            heigth, width = slice.shape
            slice = slice.view(1, heigth, width)
            slice = self.augmentor.normalize(slice)
            slice = slice.view(heigth, width)
            normalized_image.append(slice)
        image = torch.stack(normalized_image)

        return image, coords_n_scores

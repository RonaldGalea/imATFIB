import torch
import cv2
import numpy as np
from pathlib import Path

import constants
from data_loading import dataset_base
from utils.dataset_utils import reading
from utils.training_utils import box_utils
from utils.training_utils import training_processing


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
            image, mask, info = reading.read_img_mask_pair(image_path=path,
                                                           dset_name=self.dset_name,
                                                           seg_type=self.seg_type)
            self.images.append(image)
            self.masks.append(mask)
            self.infos.append(info)

    def normalize(self, image):
        """
        Args:
        image: torch.tensor (D H W)
        Normalizes validation volume
        """
        mean_std = None
        if self.params.norm_type == constants.per_dataset:
            mean_std = [self.dataset_mean, self.dataset_std]

        image = training_processing.normalize_volume(image, mean_std)
        return image


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
        # print("Path to volume", self.paths[idx])
        image, mask, header_info = self.images[idx], self.masks[idx], self.infos[idx]

        # D x H x W
        image, mask = np.transpose(image, (2, 0, 1)), np.transpose(mask, (2, 0, 1))

        image, reconstruction_info, orig_image = self.augmentor.segmentor_valid_data(image, mask)

        # torch tensors
        mask = mask.astype(np.int64)
        mask = torch.from_numpy(mask)

        image = torch.from_numpy(image)
        image = self.normalize(image)

        if self.visualization_mode:
            return image, mask, reconstruction_info, orig_image
        return image, mask, reconstruction_info, header_info


class MRI_Dataset_3d_Detection(MRI_Dataset_3d):
    def __init__(self, dset_name, dset_type, paths, params, config):
        super(MRI_Dataset_3d_Detection, self).__init__(dset_name, dset_type, paths, params, config)
        self.anchors_xyxy = torch.tensor(params.anchors).to(
            torch.float32) / self.params.default_height
        self.anchors_xywh = box_utils.corners_to_wh(self.anchors_xyxy)

    def __getitem__(self, batched_indices):
        """
        Returns:
        torch.tensor: D x H x W (float32 image)
        torch.tensor: D x 5 + #anchors
        """
        idx = batched_indices[0]
        image, mask = self.images[idx], self.masks[idx]
        # D x H x W
        image, mask = np.transpose(image, (2, 0, 1)), np.transpose(mask, (2, 0, 1))

        image, coords_n_scores = self.augmentor.detector_valid_data(image, mask)
        coords_n_scores = torch.tensor(coords_n_scores).to(torch.float32)
        coords_n_scores[:, :4] /= self.params.default_height
        gt_for_anc = box_utils.get_gt_for_anchors(coords_n_scores[:, :4], self.anchors_xyxy)

        # concatenate the gt for anc information as well
        coords_scores_gtanc = torch.cat([coords_n_scores, gt_for_anc], dim=1)
        coords_scores_gtanc = coords_scores_gtanc.to(torch.float32)

        image = torch.from_numpy(image)
        image = self.normalize(image)

        return image, coords_scores_gtanc


class MRI_Dataset_3d_Segmentation_Test(MRI_Dataset_3d):
    """
    Dataloader for the test set
    """

    def load_everything_in_memory(self):
        """
        Since the datasets are small enough to be loaded wholely in memory....

        self.images = list of 3D volumes
        """
        # overwrite this function to not have masksss
        self.images, self.infos = [], []
        for path in self.paths:
            image, info = reading.read_image(image_path=path, type="pred")
            self.images.append(image)
            self.infos.append(info)

    def __getitem__(self, batched_indices):
        """
        Returns:
        torch.tensor: D x H x W (float32 image)
        torch.tensor: D x H x W (int64 mask)
        """
        idx = batched_indices[0]
        image, header_info = self.images[idx], self.infos[idx]

        # D x H x W
        image = np.transpose(image, (2, 0, 1))
        orig_shape = image.shape[1:]
        image = self.augmentor.resize_volume(image)

        image = torch.from_numpy(image)
        image = self.normalize(image)

        return image, orig_shape, header_info, self.paths[idx]

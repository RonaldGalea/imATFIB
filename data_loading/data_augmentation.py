import numpy as np
import torch
import cv2
from collections import namedtuple
import random
from albumentations import (
    HorizontalFlip,
    Compose,
    ElasticTransform,
    RandomGamma,
    Resize,
    Rotate
)
import torchvision.transforms.functional as F

import constants
import general_config
from utils.ROI_crop import roi_crop
from utils import visualization


Reconstruction_Info = namedtuple('Reconstruction_Info', ['orig_roi', 'orig_pred'])


class Augmentor():
    """
    Handles data augmentation
    """
    def __init__(self, params):
        self.params = params
        self.resize = Resize(height=params.default_height, width=params.default_width,
                             interpolation=params.interpolation)
        self.aug = Compose([
            Rotate(limit=15),
            HorizontalFlip(p=0.05)])

    def augment_data(self, image, mask):
        augmented = self.aug(image=image, mask=mask)
        return augmented['image'], augmented['mask']

    def normalize(self, image):
        """
        Args:
        image: torch.tensor (1 H W)
        """
        if self.params.norm_type == constants.per_dataset:
            image = F.normalize(image, [general_config.dataset_mean], [general_config.dataset_std])
        elif self.params.norm_type == constants.per_slice:
            image = F.normalize(image, [torch.mean(image)], [torch.std(image)])
        return image

    def extract_ROI(self, image, mask, idx=0):
        """
        Args:
        image: ndarray: 2D sample image from which to extract roi
        mask: ndarray: 2D label image, used for extracting relativ roi coords

        to extract relative roi from mask (as needed when training), give the mask as both args

        Extracts a regtangle part of the image, containing the area of interest (labeled area)
        The ROI might have varying size, so this function always interpolates it to the correct
        input resolution of the model
        """
        if self.params.roi_crop == constants.global_roi_extraction:
            x_max, x_min = general_config.x_roi_max, general_config.x_roi_min
            y_max, y_min = general_config.y_roi_max, general_config.y_roi_min
        else:
            x_max, x_min, y_max, y_min = roi_crop.get_mask_bounds(mask, self.params)
            x_max, x_min, y_max, y_min = self.get_minimum_size(x_max, x_min, y_max, y_min)

        # extract the roi, relative or absolute
        roi_horizontal = slice(x_min, x_max+1)
        roi_vertical = slice(y_min, y_max+1)

        roi = image[roi_vertical, roi_horizontal]

        # resize roi to input res
        roi = cv2.resize(roi, dsize=(self.params.roi_width, self.params.roi_height))

        return roi, (x_max, x_min, y_max, y_min)

    def extract_ROI_3d(self, volume, mask):
        roi_volume = []
        orig_roi_infos = []
        for i, (image_slice, mask_slice) in enumerate(zip(volume, mask)):
            roi, orig_roi_info = self.extract_ROI(image_slice, mask_slice, i)
            orig_roi_infos.append(orig_roi_info)
            # cv2.waitKey(0)
            roi_volume.append(roi)
        roi_volume = np.stack(roi_volume)

        r_info = Reconstruction_Info(orig_roi_infos, (self.params.default_height, self.params.default_width))

        return roi_volume, r_info

    def resize_volume_HW(self, image, mask):
        """
        Resizes image and mask volumes along H and W

        Returns:
        resized ndarrays
        """
        resized_image, resized_mask = [], []
        for img_slice, mask_slice in zip(image, mask):
            resized_img_slice, resized_mask_slice = self.resize_slice_HW(img_slice, mask_slice)
            resized_image.append(resized_img_slice)
            resized_mask.append(resized_mask_slice)

        return np.array(resized_image), np.array(resized_mask)

    def resize_slice_HW(self, image, mask):
        image, mask = self.resize(image=image, mask=mask).values()

        return image, mask

    def get_minimum_size(self, x_max, x_min, y_max, y_min):
        """
        In the case of relative roi extraction, resize crop such that the minimum size is 64x64
        """
        if x_max - x_min + 1 < 64:
            mid = (x_max + x_min) // 2
            x_max = mid + 32
            x_min = mid - 31
        if x_min < 0 or x_max > self.params.default_width:
            raise ValueError("Can't use a symmetric formula here, implement something else...")

        if y_max - y_min + 1 < 64:
            mid = (y_max + y_min) // 2
            y_max = mid + 32
            y_min = mid - 31
        if y_min < 0 or y_max > self.params.default_height:
            raise ValueError("Can't use a symmetric formula here, implement something else...")

        return x_max, x_min, y_max, y_min

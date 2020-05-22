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
    RandomResizedCrop,
    Rotate
)
import torchvision.transforms.functional as F

import constants
import general_config
from utils.ROI_crop import roi_crop
from utils import visualization
from experiments import general_dataset_settings


Reconstruction_Info = namedtuple('Reconstruction_Info', ['orig_roi', 'orig_pred'])


class Augmentor():
    """
    Handles data preprocessing and augmentation
    """

    def __init__(self, params):
        self.params = params
        self.plain_resize = Resize(height=params.default_height, width=params.default_width)
        self.random_crop_resize = RandomResizedCrop(height=params.default_height,
                                                    width=params.default_width,
                                                    scale=(0.25, 1.0), ratio=(0.9, 1.1))
        self.roi_resize = Resize(height=params.roi_height, width=params.roi_width)
        self.roi_random_crop_resize = RandomResizedCrop(height=params.roi_height,
                                                        width=params.roi_width,
                                                        scale=(0.7, 1.0), ratio=(0.9, 1.1))
        self.initialize_elements()

    def prepare_data_train(self, image, mask):
        """
        1. get the data to the defaul size either by simple rescaling or random resized crop
        """
        image, mask = self.resizer(image=image, mask=mask).values()
        # image, mask = self.plain_resize(image=image, mask=mask).values()

        # extract roi
        if self.params.roi_crop != constants.no_roi_extraction:
            image, _ = self.extract_ROI(image, mask)
            mask, _ = self.extract_ROI(mask, mask)

        # apply augmentation, efficiently only on the smaller roi if the case
        if self.params.data_augmentation != constants.no_augmentation:
            image, mask = self.aug(image=image, mask=mask).values()

        # if roi was used, interpolate to roi size
        if self.params.roi_crop != constants.no_roi_extraction:
            if self.params.data_augmentation != constants.no_augmentation:
                # image, mask = self.roi_random_crop_resize(image=image, mask=mask).values()
                # i originally intended to use random resized crop, but it just performs bad, so for now stays commented
                image, mask = self.roi_resize(image=image, mask=mask).values()
            else:
                image, mask = self.roi_resize(image=image, mask=mask).values()

        return image, mask

    def prepare_data_val(self, volume, mask):
        # resize only input to default size, the resized mask is only used for relative roi
        volume, resized_mask = self.resize_volume_HW(volume, mask)

        reconstruction_info = None
        if self.params.roi_crop != constants.no_roi_extraction:
            volume, reconstruction_info = self.extract_ROI_3d(volume, resized_mask)

        return volume, reconstruction_info

    def normalize(self, image):
        """
        Args:
        image: torch.tensor (1 H W)
        """
        if self.params.norm_type == constants.per_dataset:
            image = F.normalize(image, [self.dataset_mean], [self.dataset_std])
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
            x_max, x_min = self.x_roi_max, self.x_roi_min
            y_max, y_min = self.y_roi_max, self.y_roi_min
        else:
            x_max, x_min, y_max, y_min = roi_crop.get_mask_bounds(mask, self.params)
            x_max, x_min, y_max, y_min = self.get_minimum_size(x_max, x_min, y_max, y_min)

        # extract the roi, relative or absolute
        roi_horizontal = slice(x_min, x_max+1)
        roi_vertical = slice(y_min, y_max+1)

        roi = image[roi_vertical, roi_horizontal]

        return roi, (x_max, x_min, y_max, y_min)

    def extract_ROI_3d(self, volume, mask):
        roi_volume = []
        orig_roi_infos = []
        for i, (image_slice, mask_slice) in enumerate(zip(volume, mask)):
            roi, orig_roi_info = self.extract_ROI(image_slice, mask_slice, i)
            orig_roi_infos.append(orig_roi_info)

            # resize roi to input res
            roi = cv2.resize(roi, dsize=(self.params.roi_width, self.params.roi_height))
            roi_volume.append(roi)
        roi_volume = np.stack(roi_volume)

        r_info = Reconstruction_Info(
            orig_roi_infos, (self.params.default_height, self.params.default_width))

        return roi_volume, r_info

    def resize_volume_HW(self, image, mask):
        """
        Resizes image and mask volumes along H and W

        Returns:
        resized ndarrays
        """
        resized_image, resized_mask = [], []
        for img_slice, mask_slice in zip(image, mask):
            resized_img_slice, resized_mask_slice = self.plain_resize(image=img_slice,
                                                                      mask=mask_slice).values()
            resized_image.append(resized_img_slice)
            resized_mask.append(resized_mask_slice)

        return np.array(resized_image), np.array(resized_mask)

    def get_minimum_size(self, x_max, x_min, y_max, y_min):
        """
        In the case of relative roi extraction, resize crop such that the minimum size is
        default height/width // 4
        """
        width = self.params.default_width
        if x_max - x_min + 1 < width // 4:
            mid = (x_max + x_min) // 2
            x_max = mid + width // 8
            x_min = mid - (width // 8) - 1
        if x_min < 0 or x_max > width:
            print("Computed xmin xmax: ", x_min, x_max)
            raise ValueError("Can't use a symmetric formula here, implement something else...")

        height = self.params.default_height
        if y_max - y_min + 1 < height // 4:
            mid = (y_max + y_min) // 2
            y_max = mid + height // 8
            y_min = mid - (height // 8) - 1
        if y_min < 0 or y_max > height:
            raise ValueError("Can't use a symmetric formula here, implement something else...")

        return x_max, x_min, y_max, y_min

    def initialize_elements(self):
        # use random crop if not using roi extraction
        if self.params.data_augmentation != constants.no_augmentation and self.params.roi_crop == constants.no_roi_extraction:
            self.resizer = self.random_crop_resize
        else:
            self.resizer = self.plain_resize

        starting_aug = [
            Rotate(limit=15),
            HorizontalFlip(p=0.5)]

        heavy_aug = [
            ElasticTransform(p=0.1)
        ]

        if self.params.data_augmentation == constants.heavy_augmentation:
            raise ValueError("Heavy augmentation not supported yet")
            # starting_aug.extend(heavy_aug)
        self.aug = Compose(starting_aug)

        if self.params.dataset == constants.imatfib_root_dir:
            self.dataset_mean = general_dataset_settings.imatfib_dataset_mean
            self.dataset_std = general_dataset_settings.imatfib_dataset_std

            self.x_roi_max = general_dataset_settings.imatfib_x_roi_max
            self.x_roi_min = general_dataset_settings.imatfib_x_roi_min
            self.y_roi_max = general_dataset_settings.imatfib_y_roi_max
            self.y_roi_min = general_dataset_settings.imatfib_y_roi_min

        elif self.params.dataset == constants.acdc_root_dir:
            self.dataset_mean = general_dataset_settings.acdc_dataset_mean
            self.dataset_std = general_dataset_settings.acdc_dataset_std

        else:
            raise NotImplementedError("Haven't gotten to mmwhs yet..")

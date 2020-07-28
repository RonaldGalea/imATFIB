import numpy as np
import torch
import cv2
from albumentations import (
    HorizontalFlip,
    Compose,
    ElasticTransform,
    Resize,
    RandomResizedCrop,
    Rotate,
    GaussNoise,
    GaussianBlur
)
import torchvision.transforms.functional as F

import constants
from utils.ROI_crop import roi_crop
import general_dataset_settings


class Augmentor():
    """
    Handles data preprocessing and augmentation
    """

    def __init__(self, params, config):
        self.params = params
        self.config = config
        self.plain_resize = Resize(height=params.default_height, width=params.default_width)
        self.random_crop_resize = RandomResizedCrop(height=params.default_height,
                                                    width=params.default_width,
                                                    scale=params.random_crop_scale,
                                                    ratio=params.random_crop_ratio)
        self.roi_resize = Resize(height=params.roi_height, width=params.roi_width)
        self.initialize_elements()

    def segmentor_train_data(self, image, mask):
        image, mask = self.resizer(image=image, mask=mask).values()
        # extract roi
        if self.params.roi_crop != constants.no_roi_extraction:
            box_coords = roi_crop.compute_ROI_coords(mask, self.params, self.config)
            image = roi_crop.extract_ROI(image, box_coords)
            mask = roi_crop.extract_ROI(mask, box_coords)

        # apply augmentation, efficiently only on the smaller roi if the case
        if self.params.data_augmentation != constants.no_augmentation:
            image, mask = self.aug(image=image, mask=mask).values()

        # if roi was used, interpolate to roi size
        if self.params.roi_crop != constants.no_roi_extraction:
            image, mask = self.roi_resize(image=image, mask=mask).values()

        return image, mask

    def detector_train_data(self, image, mask):
        image, mask = self.resizer(image=image, mask=mask).values()

        if self.params.data_augmentation != constants.no_augmentation:
            image, mask = self.aug(image=image, mask=mask).values()
        # need to return the ROI coords and the binary value whether the heart is present
        (x_min, y_min, x_max, y_max) = roi_crop.compute_ROI_coords(mask, self.params, self.config)
        score = self.no_roi_check(x_min, y_min, x_max, y_max)
        return image, (x_min, y_min, x_max, y_max, score)

    def segmentor_valid_data(self, volume, mask):
        reconstruction_info = None
        volume, resized_mask = self.resize_volume_HW(volume, mask)
        if self.params.roi_crop != constants.no_roi_extraction:
            volume, reconstruction_info = roi_crop.extract_ROI_3d(volume, resized_mask, self.params, self.config)
        return volume, reconstruction_info

    def detector_valid_data(self, volume, mask):
        volume, resized_mask = self.resize_volume_HW(volume, mask)
        coords_n_scores = []
        for slice in resized_mask:
            (x_min, y_min, x_max, y_max) = roi_crop.compute_ROI_coords(slice, self.params, self.config, True)
            score = self.no_roi_check(x_min, y_min, x_max, y_max)
            coords_n_scores.append((x_min, y_min, x_max, y_max, score))

        return volume, coords_n_scores

    def normalize(self, image):
        """
        Args:
        image: torch.tensor (1 H W)
        """
        if self.params.norm_type == constants.per_dataset:
            image = F.normalize(image, [self.dataset_mean], [self.dataset_std])
        elif self.params.norm_type == constants.per_slice:
            std = torch.std(image)
            if std == 0:
                std = 1
            image = F.normalize(image, [torch.mean(image)], [std])
        return image

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

    def initialize_elements(self):
        self.resizer = self.plain_resize
        # use random crop if not using roi extraction, unless otherwise specified by scale
        if self.params.data_augmentation != constants.no_augmentation and hasattr(self.params, "roi_crop"):
            if self.params.random_crop_scale != [1, 1]:
                print("Using random resized cropping")
                self.resizer = self.random_crop_resize

        starting_aug = [
            Rotate(limit=15),
            HorizontalFlip(p=0.5)]

        heavy_aug = [
            # RandomGamma(p=0.1),
            ElasticTransform(p=0.1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GaussNoise(p=0.05),
            GaussianBlur(p=0.05)
        ]

        if self.params.data_augmentation == constants.heavy_augmentation:
            starting_aug.extend(heavy_aug)
        self.aug = Compose(starting_aug)

        if self.config.dataset == constants.imatfib_root_dir:
            self.dataset_mean = general_dataset_settings.imatfib_dataset_mean
            self.dataset_std = general_dataset_settings.imatfib_dataset_std

        elif self.config.dataset == constants.acdc_root_dir:
            self.dataset_mean = general_dataset_settings.acdc_dataset_mean
            self.dataset_std = general_dataset_settings.acdc_dataset_std

        else:
            self.dataset_mean = general_dataset_settings.mmwhs_dataset_mean
            self.dataset_std = general_dataset_settings.mmwhs_dataset_std

        if self.params.norm_type == constants.per_slice:
            print("Using per slice normalization!")
        else:
            print("Dataset mean and std: ", self.dataset_mean, self.dataset_std)

    def no_roi_check(self, x_min, y_min, x_max, y_max):
        if (0, 0, self.params.default_width - 1, self.params.default_height - 1) == (x_min, y_min, x_max, y_max):
            return 0
        return 1

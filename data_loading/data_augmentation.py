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

from utils.ROI_crop import roi_crop
import constants
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
            setup = roi_crop.get_roi_crop_setup(self.params, self.config)
            box_coords = roi_crop.compute_ROI_coords(mask, self.params, setup)
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
        setup = roi_crop.get_roi_crop_setup(self.params, self.config)
        (x_min, y_min, x_max, y_max) = roi_crop.compute_ROI_coords(mask, self.params, setup)
        score = roi_crop.no_roi_check(x_min, y_min, x_max, y_max, self.params)
        return image, (x_min, y_min, x_max, y_max, score)

    def segmentor_valid_data(self, volume, mask):
        reconstruction_info = None
        resized_volume, resized_mask = self.resize_volume_mask_pair(volume, mask)
        if self.params.roi_crop != constants.no_roi_extraction:
            volume_roi, reconstruction_info = roi_crop.extract_ROI_3d(resized_volume, resized_mask, self.params, self.config)
            return volume_roi, reconstruction_info, resized_volume

        return resized_volume, reconstruction_info, resized_volume

    def detector_valid_data(self, volume, mask):
        volume, resized_mask = self.resize_volume_mask_pair(volume, mask)
        coords_n_scores = roi_crop.get_volume_coords(resized_mask, self.params, self.config, True)

        return volume, coords_n_scores

    def resize_volume_mask_pair(self, image, mask):
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

    def resize_volume(self, image):
        """
        Resizes image volumes along H and W

        Returns:
        resized ndarrays
        """
        resized_image = []
        for img_slice in image:
            resized_img_slice = self.plain_resize(image=img_slice)['image']
            resized_image.append(resized_img_slice)

        return np.array(resized_image)

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

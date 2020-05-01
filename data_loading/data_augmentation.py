import numpy as np
import torch
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


class Augmentor():
    """
    Handles data augmentation
    """
    def __init__(self, params):
        self.params = params
        self.resize = Resize(height=params.height, width=params.width,
                             interpolation=params.interpolation)
        self.aug = Compose([
            Rotate(limit=15),
            HorizontalFlip(p=0.05)])

    def resize_volume_HW(self, image, mask):
        """
        Resizes image and mask volumes along H and W

        Returns:
        resized ndarrays
        """
        resized_image, resized_mask = [], []
        for img_slice, mask_slice in zip(image, mask):
            resized_img_slice, resized_mask_slice = self.resize(image=img_slice,
                                                                mask=mask_slice).values()
            resized_image.append(resized_img_slice)
            resized_mask.append(resized_mask_slice)

        return np.array(resized_image), np.array(resized_mask)

    def resize_slice_HW(self, image, mask):
        image, mask = self.resize(image=image, mask=mask).values()

        return image, mask

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

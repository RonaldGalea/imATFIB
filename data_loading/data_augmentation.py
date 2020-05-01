import numpy as np
from albumentations import (
    Resize
)


class Augmentor():
    """
    Handles data augmentation
    """
    def __init__(self, params):
        self.params = params
        self.resize = Resize(height=params.height, width=params.width,
                             interpolation=params.interpolation)

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

    def resize_image_only(self, image):
        resized_image = []
        for img_slice in image:
            resized_img_slice = self.resize(image=img_slice)['image']
            resized_image.append(resized_img_slice)

        return np.array(resized_image)

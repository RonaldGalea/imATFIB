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
from experiments import general_dataset_settings


class Augmentor():
    """
    Handles data preprocessing and augmentation
    """

    def __init__(self, params):
        self.params = params
        self.plain_resize = Resize(height=params.default_height, width=params.default_width)
        self.random_crop_resize = RandomResizedCrop(height=params.default_height,
                                                    width=params.default_width,
                                                    scale=params.random_crop_scale,
                                                    ratio=params.random_crop_ratio)
        self.roi_resize = Resize(height=params.roi_height, width=params.roi_width)
        self.initialize_elements()

    def prepare_data_train(self, image, mask):
        """
        1. get the data to the defaul size either by simple rescaling or random resized crop
        """
        image, mask = self.resizer(image=image, mask=mask).values()

        # extract roi
        if self.params.roi_crop != constants.no_roi_extraction:
            box_coords = self.compute_ROI_coords(mask)
            image = self.extract_ROI(image, box_coords, type="image")
            mask = self.extract_ROI(mask, box_coords, type="mask")

        # apply augmentation, efficiently only on the smaller roi if the case
        if self.params.data_augmentation != constants.no_augmentation:
            image, mask = self.aug(image=image, mask=mask).values()

        # if roi was used, interpolate to roi size
        if self.params.roi_crop != constants.no_roi_extraction:
            if self.params.data_augmentation != constants.no_augmentation:
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

    def compute_ROI_coords(self, mask, validation=False):
        """
        mask: ndarray: 2D label image, used for extracting relativ roi coords
        validation: bool: relatvie roi extraction differs

        computes bounds of labelled area

        returns: tuple of box coords
        """
        if self.params.roi_crop == constants.global_roi_extraction:
            x_max, x_min = self.x_roi_max, self.x_roi_min
            y_max, y_min = self.y_roi_max, self.y_roi_min
        else:
            x_max, x_min, y_max, y_min = roi_crop.get_mask_bounds(mask, self.params)
            x_max, x_min, y_max, y_min = self.get_minimum_size(
                x_max, x_min, y_max, y_min, validation)

        return (x_max, x_min, y_max, y_min)

    def extract_ROI(self, image, box_coords, type="mask"):
        """
        Args:
        image: ndarray: 2D sample image from which to extract roi
        mask: ndarray: 2D label image, used for extracting relativ roi coords
        validation: bool: relatvie roi extraction differs

        to extract relative roi from mask (as needed when training), give the mask as both args

        Extracts a regtangle part of the image, containing the area of interest (labeled area)
        The ROI might have varying size, so this function always interpolates it to the correct
        input resolution of the model
        """
        x_max, x_min, y_max, y_min = box_coords

        # extract the roi, relative or absolute
        roi_horizontal = slice(x_min, x_max+1)
        roi_vertical = slice(y_min, y_max+1)

        roi = image[roi_vertical, roi_horizontal]

        return roi

    def extract_ROI_3d(self, volume, mask):
        roi_volume = []
        orig_roi_infos = []
        for i, (image_slice, mask_slice) in enumerate(zip(volume, mask)):
            box_coords = self.compute_ROI_coords(mask_slice, validation=True)
            roi = self.extract_ROI(image_slice, box_coords)
            orig_roi_infos.append(box_coords)

            # resize roi to input res
            roi = cv2.resize(roi, dsize=(self.params.roi_width, self.params.roi_height))
            roi_volume.append(roi)
        roi_volume = np.stack(roi_volume)

        return roi_volume, orig_roi_infos

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

    def get_minimum_size(self, x_max, x_min, y_max, y_min, validation=False):
        """
        In the case of relative roi extraction, resize crop such that the minimum size is
        default height/width // 4
        """
        # practically, region cut won't always be perfect, so add a perturbation value
        if self.params.relative_roi_perturbation:
            perfect_roi_width, perfect_roi_height = x_max - x_min, y_max - y_min
            width_perturb_limit = perfect_roi_width // 6
            height_perturb_limit = perfect_roi_height // 6

            if not validation:
                # perturb up to 33% of original size
                x_min_perturb = np.random.randint(0, width_perturb_limit)
                x_max_perturb = np.random.randint(0, width_perturb_limit)
                y_max_perturb = np.random.randint(0, height_perturb_limit)
                y_min_perturb = np.random.randint(0, height_perturb_limit)
            else:
                # if we're validating, add fixed perturbation to avoid a lucky eval
                x_min_perturb = width_perturb_limit // 2
                x_max_perturb = width_perturb_limit // 2
                y_max_perturb = height_perturb_limit // 2
                y_min_perturb = height_perturb_limit // 2

            x_min -= x_min_perturb
            x_max += x_max_perturb
            y_min -= y_min_perturb
            y_max += y_max_perturb

        # clamp values back to image range
        width = self.params.default_width
        height = self.params.default_height

        x_min = max(x_min, 0)
        x_max = min(x_max, width - 1)
        y_min = max(y_min, 0)
        y_max = min(y_max, height - 1)

        if x_max - x_min + 1 < width // 4:
            mid = (x_max + x_min) // 2
            x_max = mid + width // 8
            x_min = mid - (width // 8) - 1

        if y_max - y_min + 1 < height // 4:
            mid = (y_max + y_min) // 2
            y_max = mid + height // 8
            y_min = mid - (height // 8) - 1

        if x_min < 0 or x_max > width:
            print("Computed xmin xmax: ", x_min, x_max)
            raise ValueError("Can't use a symmetric formula here, implement something else...")

        if y_min < 0 or y_max > height:
            print("Computed y_max, y_min: ", y_max, y_min)
            raise ValueError("Can't use a symmetric formula here, implement something else...")

        return x_max, x_min, y_max, y_min

    def initialize_elements(self):
        self.resizer = self.plain_resize
        # use random crop if not using roi extraction, unless otherwise specified by scale
        if self.params.data_augmentation != constants.no_augmentation and self.params.roi_crop == constants.no_roi_extraction:
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

        print("Dataset mean and std: ", self.dataset_mean, self.dataset_std)

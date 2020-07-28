import torch
import numpy as np

from data_loading import dataset_base
from utils.dataset_utils import reading


class MRI_Dataset_2d(dataset_base.MRI_Dataset):
    """
    Class to handle data loading
    Receives a list of paths to the input samples, and finds the corresponding ground truth masks
    depending on the current directory structure
    """

    def __init__(self, dset_name, dset_type, paths, params, config):
        super(MRI_Dataset_2d, self).__init__(dset_name, dset_type, paths, params, config)

    def __len__(self):
        return len(self.images)

    def load_everything_in_memory(self):
        """
        Since the datasets are small enough to be loaded wholely in memory....

        self.images = list of 2D slices
        self.masks = labels for those slices
        """
        images, masks = [], []
        for path in self.paths:
            image, mask, info = reading.get_img_mask_pair(image_path=path,
                                                          dset_name=self.dset_name,
                                                          seg_type=self.seg_type)
            depth = image.shape[2]
            for i in range(depth):
                images.append(image[:, :, i])
                masks.append(mask[:, :, i])

        self.images = images
        self.masks = masks

    def normalize_image(self, image):
        height, width = image.shape
        image = image.view(1, height, width)
        image = self.augmentor.normalize(image)
        return image

    def get_images(self):
        return self.images


class MRI_Dataset_2d_Segmentation(MRI_Dataset_2d):
    def __init__(self, dset_name, dset_type, paths, params, config):
        super(MRI_Dataset_2d_Segmentation, self).__init__(
            dset_name, dset_type, paths, params, config)

    def __getitem__(self, batched_indices):
        """
        Returns:
        - torch.tensor of shape: Batch x 1 x H x W (image: float32)
        - torch.tensor of shape: Batch x H x W (mask: int64)
        """
        images_list, masks_list = [], []
        for idx in batched_indices:
            image, mask = self.images[idx], self.masks[idx]
            image, mask = self.augmentor.segmentor_train_data(image, mask)

            # torch tensors
            mask = mask.astype(np.int64)
            mask = torch.from_numpy(mask)

            masks_list.append(mask)

            image = torch.from_numpy(image)
            image = self.normalize_image(image)
            images_list.append(image)

        return torch.stack(images_list), torch.stack(masks_list)


class MRI_Dataset_2d_Detection(MRI_Dataset_2d):
    def __init__(self, dset_name, dset_type, paths, params, config):
        super(MRI_Dataset_2d_Detection, self).__init__(dset_name, dset_type, paths, params, config)

    def __getitem__(self, batched_indices):
        """
        Returns:
        - torch.tensor of shape: Batch x 1 x H x W (image: float32)
        - torch.tensor of shape: Batch x 5 (mask: ROI coords and heart heart_presence)
        """
        images_list = []
        coords_n_scores = []
        for idx in batched_indices:
            image, mask = self.images[idx], self.masks[idx]
            image, coords_n_score = self.augmentor.detector_train_data(image, mask)

            coords_n_score = torch.tensor(coords_n_score)
            coords_n_score = coords_n_score.to(torch.float32)

            coords_n_scores.append(coords_n_score)

            image = torch.from_numpy(image)
            image = self.normalize_image(image)
            images_list.append(image)

        return torch.stack(images_list), torch.stack(coords_n_scores)

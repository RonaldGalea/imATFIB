import torch

from data_loading import dataset_base
from utils.dataset_utils import data_normalization


class MRI_Dataset_3d(dataset_base.MRI_Dataset):
    """
    Class to handle data for the validation set

    Returns a volume at a time, with untouched masks, for correct evaluation
    """
    def __init__(self, dset_name, dset_type, paths, params):
        super(MRI_Dataset_3d, self).__init__(dset_name, dset_type, paths, params)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, batched_indices):
        """
        Returns:
        torch.tensor: D x H x W (float32 image)
        torch.tensor: D x H x W (int64 mask)
        """
        idx = batched_indices[0]
        image, mask, info = self.images[idx], self.masks[idx], self.infos[0]

        # torch tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        mask = mask.to(torch.int64)

        return image, mask, info

    def necessary_preprocessing(self):
        for i, image in enumerate(self.images):
            resized_image = self.augmentor.resize_image_only(image)
            self.images[i] = resized_image

        data_normalization.normalize(self.images, self.norm_type)

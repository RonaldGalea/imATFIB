import torch
import numpy as np

from utils import statistics
from data_loading import dataset_base


class MRI_Dataset_2d(dataset_base.MRI_Dataset):
    """
    Class to handle data loading
    Receives a list of paths to the input samples, and finds the corresponding ground truth masks
    depending on the current directory structure

    There are some issues with processing volumes in batches, since volumes have varying depth.

    Solutions:
    - split everything up in 2D slices
    Pro: no more dropping batches or parts of volumes (except perhaps very last), shuffling slices
    completely may acts as a regularizing effect
    Con: ? may lead to failure to learn any inter slice dependencies (are there any for 2D though?)

    - get a an amount (batch) of volumes s.t. their total depth has at least, say, 10x batch_size. Then, if
    split batches are allowed, then they only occur every 10th batch. If no split batches are allowed,
    then some slices are dropped. However, since volumes are shuffled every epoch, there should
    be no bias as to which volume's slices are lost. The only issue with this would be the exact
    training time, since actually it's slightly less slices than the total with each epoch.

    The batch_size of volumes means how many volumes are necessary to have a depth of
    10 x slice_batch_size, and may differ by dataset

    This feels way too hacky and confusing, I will just write more bugs than words if I try this
    """
    def __init__(self, dset_name, dset_type, paths, params):
        super(MRI_Dataset_2d, self).__init__(dset_name, dset_type, paths, params)
        self.concatenate_everything()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, batched_indices):
        """
        Returns batch of slices
        """
        images_list, masks_list, infos_list = [], [], []
        for idx in batched_indices:
            image, mask, info = self.images[idx], self.masks[idx], self.infos[0]

            # torch tensors
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)

            images_list.append(image)
            masks_list.append(mask)
            infos_list.append(info)

        return torch.stack(images_list), torch.stack(masks_list), infos_list

    def necessary_preprocessing(self):
        for i, (image, mask) in enumerate(zip(self.images, self.masks)):
            resized_image, resized_mask = self.augmentor.resize_volume_HW(image, mask)
            self.images[i] = resized_image
            self.masks[i] = resized_mask

        statistics.normalize(self.images, self.norm_type)

    def concatenate_everything(self):
        self.images = np.concatenate(self.images)
        self.masks = np.concatenate(self.masks)

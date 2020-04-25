import torch
import numpy as np
from torch.utils.data import Dataset

import general_config
from utils import reading, visualization, statistics
from data_loading import data_augmentation


class MRI_Dataset(Dataset):
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
        """
        Args:
        dset_name - string: name of the dataset
        dset_type - string: train or val
        paths - list of Path: paths to data samples
        params - params.json of current experiment
        """
        self.dset_name = dset_name
        self.dset_type = dset_type
        self.paths = paths
        self.seg_type = params.seg_type
        self.norm_type = params.norm_type
        self.augmentor = data_augmentation.Augmentor(params)
        self.load_everything_in_memory()
        if general_config.visualize_dataset:
            self.visualize_dataset_samples()
        self.necessary_preprocessing()
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

    def load_everything_in_memory(self):
        """
        Since the datasets are small enough to be loaded wholely in memory....

        self.images = list of volumes
        self.masks = labels for those volumes
        self.info = list of namedtuple(affine, header) for each volume
        """
        images, masks, infos = [], [], []
        for path in self.paths:
            image, mask, info = reading.get_img_mask_pair(image_path=path,
                                                          numpy=general_config.read_numpy,
                                                          dset_name=self.dset_name,
                                                          seg_type=self.seg_type)
            # permute dimensions to D x H x W for easier indexing
            # after processing will be switched back
            images.append(np.transpose(image, (2, 0, 1)))
            masks.append(np.transpose(mask, (2, 0, 1)))
            infos.append(info)

        self.images = images
        self.masks = masks
        self.infos = infos

    def necessary_preprocessing(self):
        for i, (image, mask) in enumerate(zip(self.images, self.masks)):
            resized_image, resized_mask = self.augmentor.resize_volume_HW(image, mask)
            self.images[i] = resized_image
            self.masks[i] = resized_mask

        statistics.normalize(self.images, self.norm_type)

    def concatenate_everything(self):
        self.images = np.concatenate(self.images)
        self.masks = np.concatenate(self.masks)

    def visualize_dataset_samples(self):
        for image, mask in zip(self.images, self.masks):
            visualization.visualize_img_mask_pair(np.transpose(image, (1, 2, 0)),
                                                  np.transpose(mask, (1, 2, 0)))
            exit = input("exit? y/n")
            if exit == 'y':
                return

    def dataset_info(self):
        print("Dataset name: ", self.dset_name, "\n")
        print("Dataset type: ", self.dset_type, "\n")
        print("Elements in dataset: ", len(self.paths), "\n")
        print("Segmentation type: ", self.seg_type, "\n")

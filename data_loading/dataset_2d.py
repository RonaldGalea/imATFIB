import torch

import general_config
import constants
from data_loading import dataset_base
from utils.dataset_utils import reading
from utils import visualization


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

    - get a an amount (batch) of volumes s.t. their total depth has at least, say, 10x batch_size.
    Then, if split batches are allowed, then they only occur every 10th batch.
    If no split batches are allowed,
    then some slices are dropped. However, since volumes are shuffled every epoch, there should
    be no bias as to which volume's slices are lost. The only issue with this would be the exact
    training time, since actually it's slightly less slices than the total with each epoch.

    The batch_size of volumes means how many volumes are necessary to have a depth of
    10 x slice_batch_size, and may differ by dataset

    This feels way too hacky and confusing, I will just write more bugs than words if I try this
    """
    def __init__(self, dset_name, dset_type, paths, params):
        super(MRI_Dataset_2d, self).__init__(dset_name, dset_type, paths, params)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, batched_indices):
        """
        Returns:
        - torch.tensor of shape: Batch x 1 x H x W (image: float32)
        - torch.tensor of shape: Batch x H x W (mask: int64)
        """
        images_list, masks_list = [], []
        for idx in batched_indices:
            image, mask = self.images[idx], self.masks[idx]

            # print("Before", image.shape, mask.shape, type(image), type(mask))
            # visualization.visualize_img_mask_pair_2d(image, mask, use_orig_res=True)

            image, mask = self.augmentor.resize_slice_HW(image, mask)
            if self.params.roi_crop != constants.no_roi_extraction:
                image, _ = self.augmentor.extract_ROI(image, mask)
                mask, _ = self.augmentor.extract_ROI(mask, mask)

            if self.params.data_augmentation:
                image, mask = self.augmentor.augment_data(image, mask)

            # print("After", image.shape, mask.shape, type(image), type(mask))
            # visualization.visualize_img_mask_pair_2d(image, mask, "after_img", "after_mask", use_orig_res=True)

            # torch tensors
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)

            height, width = image.shape
            image = image.view(1, height, width)
            image = self.augmentor.normalize(image)

            mask = mask.to(torch.int64)

            images_list.append(image)
            masks_list.append(mask)

        return torch.stack(images_list), torch.stack(masks_list)

    def load_everything_in_memory(self):
        """
        Since the datasets are small enough to be loaded wholely in memory....

        self.images = list of 2D slices
        self.masks = labels for those slices
        """
        images, masks = [], []
        for path in self.paths:
            image, mask, info = reading.get_img_mask_pair(image_path=path,
                                                          numpy=general_config.read_numpy,
                                                          dset_name=self.dset_name,
                                                          seg_type=self.seg_type)
            depth = image.shape[2]
            for i in range(depth):
                images.append(image[:, :, i])
                masks.append(mask[:, :, i])

        self.images = images
        self.masks = masks

    def get_images(self):
        return self.images

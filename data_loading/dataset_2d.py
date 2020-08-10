import torch
import numpy as np
import cv2

import constants
from data_loading import dataset_base
from utils.dataset_utils import reading
from utils.training_utils import box_utils
from utils import visualization
from utils.training_utils import training_processing


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
            image, mask, info = reading.read_img_mask_pair(image_path=path,
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
        mean_std = None
        if self.params.norm_type == constants.per_dataset:
            mean_std = [self.dataset_mean, self.dataset_std]
        image = training_processing.normalize_tensor(image, mean_std)
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
        self.anchors_xyxy = torch.tensor(params.anchors).to(torch.float32) / self.params.default_height
        self.anchors_xywh = box_utils.corners_to_wh(self.anchors_xyxy)

    def __getitem__(self, batched_indices):
        """
        Returns:
        - torch.tensor of shape: Batch x 1 x H x W (image: float32)
        - torch.tensor of shape: Batch x (5 + #anchors) (mask: ROI coords, heart_presence and which anchors should predict it)
        """
        images_list = []
        coords_n_scores = []
        for idx in batched_indices:
            image, mask = self.images[idx], self.masks[idx]
            image, coords_n_score = self.augmentor.detector_train_data(image, mask)

            coords_n_score = torch.tensor(coords_n_score)
            coords_n_score = coords_n_score.to(torch.float32)

            coords_n_scores.append(coords_n_score)

            # x_min, y_min, x_max, y_max = coords_n_score[:4]
            # pos = mask == 1
            # dummy_mask = np.zeros((mask.shape))
            # dummy_mask[pos] = 255.0
            #
            # dummy_mask = cv2.rectangle(dummy_mask, (x_min, y_min), (x_max, y_max), 255, 2)
            # image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), 255, 2)
            #
            # for idx, (x1, y1, x2, y2) in enumerate(self.params.anchors):
            #     print(x1, y1, x2, y2)
            #     dummy_mask = cv2.rectangle(dummy_mask, (x_min, y_min), (x_max, y_max), 255, 2)
            #     dummy_mask = cv2.rectangle(dummy_mask, (x1, y1), (x2, y2), 255, 1)
            #     image = cv2.rectangle(image, (x1, y1), (x2, y2), 255, 1)
            #
            # gt_for_anc = box_utils.get_gt_for_anchors(coords_n_score[:4].unsqueeze(0), self.anchors_xyxy)
            # print("gt_for_current_anc", gt_for_anc)
            #
            # visualization.visualize_img_mask_pair_2d(
            #     image, dummy_mask, "after_img", "after_mask", use_orig_res=True, wait=True)
            # cv2.destroyAllWindows()

            image = torch.from_numpy(image)
            image = self.normalize_image(image)
            images_list.append(image)

        coords_n_scores = torch.stack(coords_n_scores)
        coords_n_scores[:, :4] /= self.params.default_height
        gt_for_anc = box_utils.get_gt_for_anchors(coords_n_scores[:, :4], self.anchors_xyxy)
        # print("GRFORANC", gt_for_anc)

        # concatenate the gt for anc information as well
        coords_scores_gtanc = torch.cat([coords_n_scores, gt_for_anc], dim=1)

        return torch.stack(images_list), coords_scores_gtanc

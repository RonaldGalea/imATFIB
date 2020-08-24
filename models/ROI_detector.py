import torch
import torch.nn as nn


import general_config
import constants
from models.model_parts import resnet
from utils.training_utils import box_utils


"""
Classical detection did not perform well, no longer used
"""


class ROI_Detector(nn.Module):
    """
    Network to regress to bounding box coordinates and compute a probability score of the heart
    being present in the image
    """

    def __init__(self, params, config, n_channels=1):
        super(ROI_Detector, self).__init__()
        self.params = params
        self.config = config
        self.get_anchor()
        self.get_backbone(n_channels)

        self.coordinator = nn.Sequential(nn.Conv2d(self.in_channels, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Conv2d(128, 128, kernel_size=3,
                                                   stride=1, padding=1, bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         nn.Dropout(0.1))

        self.predict_coords = nn.Linear(128, 4 * len(self.anchors_xyxy))
        self.predict_confidences = nn.Linear(128, 1 * len(self.anchors_xyxy))

        self.scorer = nn.Sequential(nn.Conv2d(self.in_channels, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Conv2d(128, 128, kernel_size=3,
                                              stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.Dropout(0.1))

        self.predict_score = nn.Linear(128, 1)

    def forward(self, x):
        """
        returns:
        ROI_coords: batch x #anchors x 4 tensor
        ROI_confs: batch x #anchors tensor
        score: batch tensor
        """
        _, features = self.backbone(x)

        coords_features = self.coordinator(features)
        coords_features = nn.functional.adaptive_avg_pool2d(
            coords_features, 1).reshape(coords_features.shape[0], -1)

        # batch x #anchors x 4
        ROI_coords = self.predict_coords(coords_features).view(features.shape[0], -1, 4)
        ROI_confs = self.predict_confidences(coords_features)

        score = self.scorer(features)
        score = nn.functional.adaptive_avg_pool2d(score, 1).reshape(score.shape[0], -1)
        score = self.predict_score(score)

        return ROI_coords, ROI_confs, score.squeeze(1)

    def get_anchor(self):
        self.anchors_xyxy = (torch.tensor(self.params.anchors, dtype=torch.float32) / self.params.default_height).to(general_config.device)
        self.anchors_xywh = box_utils.corners_to_wh(self.anchors_xyxy)

    def get_backbone(self, n_channels):
        if self.config.model_id == constants.resnet18_detector:
            self.backbone = resnet.resnet18(initial_channels=n_channels,
                                            shrinking_factor=self.params.shrinking_factor)
            self.in_channels = int(512 / self.params.shrinking_factor)
        elif self.config.model_id == constants.resnet50_detector:
            self.backbone = resnet.resnext50_32x4d(initial_channels=n_channels,
                                                   replace_stride_with_dilation=self.params.replace_stride,
                                                   shrinking_factor=self.params.shrinking_factor,
                                                   layer_count=self.params.layer_count)
            self.in_channels = int(2048 / self.params.shrinking_factor)

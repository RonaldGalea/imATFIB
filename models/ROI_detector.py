import torch
import torch.nn as nn


import general_config
import constants
from experiments import general_dataset_settings
from models.model_parts import mobilenetv2, resnet
from utils.training_utils import box_utils


class ROI_Detector(nn.Module):
    """
    Network to regress to bounding box coordinates and compute a probability score of the heart
    being present in the image
    """

    def __init__(self, params, n_channels=1):
        super(ROI_Detector, self).__init__()
        self.params = params
        self.get_anchor()
        self.get_backbone(n_channels)

        self.regressor = nn.Sequential(nn.Conv2d(self.in_channels, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(128, 128, kernel_size=3,
                                                 stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.1))

        self.predict_coords = nn.Linear(128, 4)

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
        _, features = self.backbone(x)

        ROI_coords = self.regressor(features)
        ROI_coords = nn.functional.adaptive_avg_pool2d(
            ROI_coords, 1).reshape(ROI_coords.shape[0], -1)

        ROI_coords = self.predict_coords(ROI_coords)

        score = self.scorer(features)
        score = nn.functional.adaptive_avg_pool2d(score, 1).reshape(score.shape[0], -1)
        score = self.predict_score(score)

        return ROI_coords, score

    def get_anchor(self):
        """
        Will be the global encompassing box on the training set
        Anchor will be in center_x, center_y, width, height format
        """
        if self.params.dataset == constants.imatfib_root_dir:
            self.anchor = torch.tensor(general_dataset_settings.imatfib_anchor)
        elif self.params.dataset == constants.acdc_root_dir:
            self.anchor = torch.tensor(general_dataset_settings.acdc_anchor)
        self.anchor = self.anchor.unsqueeze(0)

        # normalize coords in 0 - 1 range
        self.anchor = self.anchor / self.params.default_height
        self.anchor = box_utils.corners_to_wh(self.anchor).to(general_config.device)

    def get_backbone(self, n_channels):
        if self.params.model_id == constants.resnet18_detector:
            self.backbone = resnet.resnet18(initial_channels=n_channels,
                                            shrinking_factor=self.params.shrinking_factor)
            self.in_channels = int(512 / self.params.shrinking_factor)
        elif self.params.model_id == constants.mobilenet_detector:
            self.backbone = mobilenetv2.MobileNetV2(initial_channels=n_channels)
            self.in_channels = 320
        elif self.params.model_id == constants.resnet50_detector:
            self.backbone = resnet.resnext50_32x4d(initial_channels=n_channels,
                                                   replace_stride_with_dilation=self.params.replace_stride,
                                                   shrinking_factor=self.params.shrinking_factor,
                                                   layer_count=self.params.layer_count)
            self.in_channels = int(2048 / self.params.shrinking_factor)

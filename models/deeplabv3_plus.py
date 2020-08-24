import torch
import torch.nn as nn
import torch.nn.functional as F

import constants
from models.model_parts import mobilenetv2, aspp, resnet


class DeepLabV3_plus_base(nn.Module):
    def __init__(self, n_channels, n_classes, params, config):
        super(DeepLabV3_plus_base, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.params = params
        self.config = config
        self.softmax_layer = nn.LogSoftmax(dim=1)

    def forward(self, x):
        inter, x = self.backbone(x)
        low_res = self.low_res_feature_conv(x)
        upsampled = F.interpolate(low_res, size=inter.size()[2:],
                                  mode='bilinear', align_corners=True)
        high_res = self.high_res_feature_conv(inter)

        concat = torch.cat((upsampled, high_res), dim=1)
        logits = self.classifier(concat)
        logits = F.interpolate(logits, scale_factor=4,
                               mode='bilinear', align_corners=True)
        return self.softmax_layer(logits)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def get_low_res_feature_conv(self):
        in_channels = int(
            2048 / self.params.shrinking_factor) if self.config.model_id == constants.resnext_deeplab else 320
        if self.params.use_aspp:
            low_conv = aspp.ASPP(in_channels=in_channels)
        else:
            low_conv = mobilenetv2.ConvBNReLU(in_planes=in_channels,
                                              out_planes=int(256 / self.params.shrinking_factor),
                                              kernel_size=1, bias=False)
        return low_conv

    def get_high_res_feature_conv(self):
        if self.config.model_id == constants.resnext_deeplab:
            high_conv = mobilenetv2.ConvBNReLU(in_planes=int(256 / self.params.shrinking_factor),
                                               out_planes=int(48 / self.params.shrinking_factor),
                                               kernel_size=1, bias=False)
        elif self.config.model_id == constants.deeplab:
            high_conv = mobilenetv2.ConvBNReLU(in_planes=144, out_planes=48,
                                               kernel_size=1, bias=False)
        return high_conv

    def get_classifier(self):
        if self.config.model_id == constants.resnext_deeplab:
            in_channels, out_channels = int(
                304 / self.params.shrinking_factor), int(256 / self.params.shrinking_factor)
            classifier = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                                 stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(out_channels, self.n_classes, kernel_size=1, stride=1))

        elif self.config.model_id == constants.deeplab:
            classifier = nn.Sequential(
                mobilenetv2.ConvBNReLU(in_planes=304, out_planes=256,
                                       kernel_size=3, bias=False),
                nn.Conv2d(in_channels=256, out_channels=self.n_classes, kernel_size=1)
            )

        return classifier

    def get_backbone(self):
        if self.config.model_id == constants.resnext_deeplab:
            backbone = resnet.resnext50_32x4d(initial_channels=self.n_channels,
                                              replace_stride_with_dilation=self.params.replace_stride,
                                              shrinking_factor=self.params.shrinking_factor,
                                              layer_count=self.params.layer_count)
        elif self.config.model_id == constants.deeplab:
            backbone = mobilenetv2.MobileNetV2(initial_channels=self.n_channels)
        return backbone


class DeepLabV3_plus(DeepLabV3_plus_base):
    def __init__(self, n_channels, n_classes, params, config):
        super(DeepLabV3_plus, self).__init__(n_channels, n_classes, params, config)

        self.backbone = self.get_backbone()
        self.low_res_feature_conv = self.get_low_res_feature_conv()
        self.high_res_feature_conv = self.get_high_res_feature_conv()
        self.classifier = self.get_classifier()

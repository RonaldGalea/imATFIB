import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_parts import mobilenetv2


class DeepLabV3_plus(nn.Module):
    def __init__(self, n_channels, n_classes, use_aspp=False):
        super(DeepLabV3_plus, self).__init__()

        self.n_classes = n_classes
        self.use_aspp = False
        self.softmax_layer = nn.LogSoftmax(dim=1)

        self.low_res_feature_conv = mobilenetv2.ConvBNReLU(in_planes=320, out_planes=256,
                                                           kernel_size=1, bias=False)
        self.high_res_feature_conv = mobilenetv2.ConvBNReLU(in_planes=144, out_planes=48,
                                                            kernel_size=1, bias=False)
        self.classifier = nn.Sequential(
            mobilenetv2.ConvBNReLU(in_planes=304, out_planes=256,
                                   kernel_size=3, bias=False),
            nn.Conv2d(in_channels=256, out_channels=n_classes, kernel_size=1)
        )

        self.backbone = mobilenetv2.MobileNetV2(initial_channels=n_channels)

        # weight initialization
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

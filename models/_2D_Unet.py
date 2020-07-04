import torch.nn as nn

from models.model_parts.unet_parts import DoubleConv, Down, Up, OutConv

"""
Implementation of Unet taken from https://github.com/milesial/Pytorch-UNet/
Unet paper: Olaf Ronneberger, Philipp Fischer, Thomas Brox: https://arxiv.org/abs/1505.04597
"""

""" Full assembly of the parts to form the complete network """


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, shrinking_factor=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.softmax_layer = nn.LogSoftmax(dim=1)

        base_channels = int(64 / shrinking_factor)

        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, (base_channels * 16) / factor)
        self.up1 = Up(base_channels * 16, (base_channels * 8) / factor, bilinear)
        self.up2 = Up(base_channels * 8, (base_channels * 4) / factor, bilinear)
        self.up3 = Up(base_channels * 4, (base_channels * 2) / factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, n_classes)

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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        predictions = self.softmax_layer(logits)
        return predictions

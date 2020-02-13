# Took help from https://github.com/jfzhang95/pytorch-deeplab-xception while writing this

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):

    def __init__(self, inchannels=2048):
        super(ASPP, self).__init__()

        paddings = [0, 4, 8, 14]
        dilations = [1, 4, 8, 14]

        self.aspp1 = self._makeConv(
            kernel=1, padding=paddings[0], dilation=dilations[0])
        self.aspp2 = self._makeConv(
            kernel=3, padding=paddings[1], dilation=dilations[1])
        self.aspp3 = self._makeConv(
            kernel=3, padding=paddings[2], dilation=dilations[2])
        self.aspp4 = self._makeConv(
            kernel=3, padding=paddings[3], dilation=dilations[3])

        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                  nn.Conv2d(inchannels, 256, 1, stride=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(),
                                  nn.Dropout2d(0.5))

        self.postProcess = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.5)
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        _, _, width, height = x4.shape

        x5 = self.pool(x)
        x5 = F.interpolate(x5, size=(width, height), mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.postProcess(x)

        return x

    def _makeConv(self, inchannels=2048, outchannels=256, kernel=3, padding=0, dilation=1):
        return nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=kernel, stride=1,
                      padding=padding, dilation=dilation),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(),
            nn.Dropout2d(0.5)
        )

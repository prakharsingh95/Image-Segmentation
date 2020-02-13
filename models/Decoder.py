# Took help from https://github.com/jfzhang95/pytorch-deeplab-xception while writing this

import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(Decoder, self).__init__()

        lowLevelChannels = 256

        self.lowLevelProcessor = nn.Sequential(
            nn.Conv2d(lowLevelChannels, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout2d(0.5)
        )

        self.postProcessor = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.15),
            nn.Conv2d(256, NUM_CLASSES, kernel_size=1, stride=1)
        )

    def forward(self, asppFeat, lowLevelFeat):
        lowLevelFeat = self.lowLevelProcessor(lowLevelFeat)

        _, _, lowLevelWidth, lowLevelHeight = lowLevelFeat.shape
        asppFeat = F.interpolate(asppFeat, size=(lowLevelWidth, lowLevelHeight), mode='bilinear', align_corners=True)

        x = torch.cat((lowLevelFeat, asppFeat), dim=1)

        x = self.postProcessor(x)

        return x
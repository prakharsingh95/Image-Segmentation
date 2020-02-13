"""
Code for ResNet-101 based largely on https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnet101

Also took help from https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/train.py
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # print(out.shape, identity.shape)
        out += identity
        out = self.relu(out)

        return out


class ResNetTrunk(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4, 23], zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetTrunk, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)

        self.layer4 = nn.Sequential(
            # self._make_layer(block, 512, 1, stride=1, dilation=2),
            block(1024, 512, stride=1, dilation=4, downsample=nn.Sequential(
                conv1x1(1024, 2048, 1),
                norm_layer(2048),
            )),
            block(2048, 512, stride=1, dilation=4),
            block(2048, 512, stride=1, dilation=4),
            # block(2048, 512, stride=1, dilation=8),
            # block(2048, 512, stride=1, dilation=16),
        )
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                                dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

        self.loadPretrained()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("Input", x.shape)
        x = self.conv1(x)
        # print("Conv1", x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print("Maxpool", x.shape)

        x = self.layer1(x)
        # print("L1", x.shape)
        lowLevelFeatures = x
        x = self.layer2(x)
        # print("L2", x.shape)
        x = self.layer3(x)
        # print("L3", x.shape)
        # print(x.shape)
        x = self.layer4(x)

        return x, lowLevelFeatures

    def loadPretrained(self):
        """https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/resnet.py
        """
        pretrain_dict = model_zoo.load_url(
            'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                # print(k)
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

# coding: utf-8

import math
import torch
from torch import nn

from .pretrained import resnet18, resnet18_pre, resnet34, resnet34_pre, resnet50, resnet50_pre
from .BasicModule import BasicModule


class PCResNet18(BasicModule):
    def __init__(self, num_classes):
        super(PCResNet18, self).__init__()

        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        self.avgpool = resnet18.avgpool
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):
        fx = self.conv1(x)
        fx = self.bn1(fx)
        fx = self.relu(fx)
        fx = self.maxpool(fx)

        fx = self.layer1(fx)
        fx = self.layer2(fx)
        fx = self.layer3(fx)
        fx = self.layer4(fx)

        fx = self.avgpool(fx)
        fx = fx.view(fx.size(0), -1)
        out_x = self.fc(fx)

        fy = self.conv1(y)
        fy = self.bn1(fy)
        fy = self.relu(fy)
        fy = self.maxpool(fy)

        fy = self.layer1(fy)
        fy = self.layer2(fy)
        fy = self.layer3(fy)
        fy = self.layer4(fy)

        fy = self.avgpool(fy)
        fy = fy.view(fy.size(0), -1)

        out_y = self.fc(fy)
        return out_x, out_y, fx, fy


class PCResNet50(BasicModule):
    def __init__(self, num_classes):
        super(PCResNet50, self).__init__()

        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        self.avgpool = resnet50.avgpool
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):
        fx = self.conv1(x)
        fx = self.bn1(fx)
        fx = self.relu(fx)
        fx = self.maxpool(fx)

        fx = self.layer1(fx)
        fx = self.layer2(fx)
        fx = self.layer3(fx)
        fx = self.layer4(fx)

        fx = self.avgpool(fx)
        fx = fx.view(fx.size(0), -1)
        out_x = self.fc(fx)

        fy = self.conv1(y)
        fy = self.bn1(fy)
        fy = self.relu(fy)
        fy = self.maxpool(fy)

        fy = self.layer1(fy)
        fy = self.layer2(fy)
        fy = self.layer3(fy)
        fy = self.layer4(fy)

        fy = self.avgpool(fy)
        fy = fy.view(fy.size(0), -1)

        out_y = self.fc(fy)
        return out_x, out_y, fx, fy

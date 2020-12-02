# coding: utf-8

import math
import copy
import torch
from torch import nn
from torch.nn import functional

from .pretrained import resnet18, resnet18_pre, resnet34, resnet34_pre, resnet50, resnet50_pre
from .BasicModule import BasicModule


class DualResNet18(BasicModule):
    def __init__(self, num_classes):
        super(DualResNet18, self).__init__()

        self.conv1_1 = resnet18.conv1
        self.bn1_1 = resnet18.bn1
        self.relu_1 = resnet18.relu
        self.maxpool_1 = resnet18.maxpool
        self.layer1_1 = resnet18.layer1
        self.layer2_1 = resnet18.layer2
        self.layer3_1 = resnet18.layer3
        self.layer4_1 = resnet18.layer4
        self.avgpool_1 = resnet18.avgpool
        self.fc_1 = nn.Linear(512, num_classes)

        self.conv1_2 = copy.deepcopy(resnet18.conv1)
        self.bn1_2 = copy.deepcopy(resnet18.bn1)
        self.relu_2 = resnet18.relu
        self.maxpool_2 = resnet18.maxpool
        self.layer1_2 = copy.deepcopy(resnet18.layer1)
        self.layer2_2 = copy.deepcopy(resnet18.layer2)
        self.layer3_2 = copy.deepcopy(resnet18.layer3)
        self.layer4_2 = copy.deepcopy(resnet18.layer4)
        self.avgpool_2 = resnet18.avgpool
        self.fc_2 = nn.Linear(512, num_classes)

        self.fc_3 = nn.Linear(1024, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):
        fx = self.conv1_1(x)
        fx = self.bn1_1(fx)
        fx = self.relu_1(fx)
        fx = self.maxpool_1(fx)

        fx = self.layer1_1(fx)
        fx = self.layer2_1(fx)
        fx = self.layer3_1(fx)
        fx = self.layer4_1(fx)

        fx = self.avgpool_1(fx)
        fx = fx.view(fx.size(0), -1)
        out_x = self.fc_1(fx)

        fy = self.conv1_2(y)
        fy = self.bn1_2(fy)
        fy = self.relu_2(fy)
        fy = self.maxpool_2(fy)

        fy = self.layer1_2(fy)
        fy = self.layer2_2(fy)
        fy = self.layer3_2(fy)
        fy = self.layer4_2(fy)

        fy = self.avgpool_2(fy)
        fy = fy.view(fy.size(0), -1)
        out_y = self.fc_2(fy)

        f_cat = torch.cat([fx, fy], 1)
        out_cat = self.fc_3(f_cat)

        return out_x, out_y, out_cat


class DualResNet50(BasicModule):
    def __init__(self, num_classes):
        super(DualResNet50, self).__init__()

        self.conv1_1 = resnet50.conv1
        self.bn1_1 = resnet50.bn1
        self.relu_1 = resnet50.relu
        self.maxpool_1 = resnet50.maxpool
        self.layer1_1 = resnet50.layer1
        self.layer2_1 = resnet50.layer2
        self.layer3_1 = resnet50.layer3
        self.layer4_1 = resnet50.layer4
        self.avgpool_1 = resnet50.avgpool
        self.fc_1 = nn.Linear(2048, num_classes)

        self.conv1_2 = copy.deepcopy(resnet50.conv1)
        self.bn1_2 = copy.deepcopy(resnet50.bn1)
        self.relu_2 = resnet50.relu
        self.maxpool_2 = resnet50.maxpool
        self.layer1_2 = copy.deepcopy(resnet50.layer1)
        self.layer2_2 = copy.deepcopy(resnet50.layer2)
        self.layer3_2 = copy.deepcopy(resnet50.layer3)
        self.layer4_2 = copy.deepcopy(resnet50.layer4)
        self.avgpool_2 = resnet50.avgpool
        self.fc_2 = nn.Linear(2048, num_classes)

        self.fc_3 = nn.Linear(2048*2, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y):
        fx = self.conv1_1(x)
        fx = self.bn1_1(fx)
        fx = self.relu_1(fx)
        fx = self.maxpool_1(fx)

        fx = self.layer1_1(fx)
        fx = self.layer2_1(fx)
        fx = self.layer3_1(fx)
        fx = self.layer4_1(fx)

        fx = self.avgpool_1(fx)
        fx = fx.view(fx.size(0), -1)
        out_x = self.fc_1(fx)

        fy = self.conv1_2(y)
        fy = self.bn1_2(fy)
        fy = self.relu_2(fy)
        fy = self.maxpool_2(fy)

        fy = self.layer1_2(fy)
        fy = self.layer2_2(fy)
        fy = self.layer3_2(fy)
        fy = self.layer4_2(fy)

        fy = self.avgpool_2(fy)
        fy = fy.view(fy.size(0), -1)
        out_y = self.fc_2(fy)

        f_cat = torch.cat([fx, fy], 1)
        out_cat = self.fc_3(f_cat)

        return out_x, out_y, out_cat

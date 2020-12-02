# coding: utf-8

import torch
from torch import nn

from .pretrained import vgg16, vgg16_pre
from .BasicModule import BasicModule


class PCVgg16(BasicModule):
    def __init__(self, num_classes):
        super(PCVgg16, self).__init__()

        self.conv1 = vgg16.features[0:5]
        self.conv2 = vgg16.features[5:10]
        self.conv3 = vgg16.features[10:17]
        self.conv4 = vgg16.features[17:24]
        self.conv5 = vgg16.features[24:31]

        self.fc1 = vgg16.classifier[0]
        self.relu1 = vgg16.classifier[1]
        self.dropout1 = vgg16.classifier[2]
        self.fc2 = vgg16.classifier[3]
        self.relu2 = vgg16.classifier[4]
        self.dropout2 = vgg16.classifier[5]
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

    def forward(self, x, y):
        fx = self.conv1(x)
        fx = self.conv2(fx)
        fx = self.conv3(fx)
        fx = self.conv4(fx)
        fx = self.conv5(fx)
        fx = fx.view(fx.size(0), -1)

        fx = self.fc1(fx)
        fx = self.fc2(self.dropout1(self.relu1(fx)))
        out_x = self.fc3(self.dropout2(self.relu2(fx)))

        fy = self.conv1(y)
        fy = self.conv2(fy)
        fy = self.conv3(fy)
        fy = self.conv4(fy)
        fy = self.conv5(fy)
        fy = fy.view(fy.size(0), -1)

        fy = self.fc1(fy)
        fy = self.fc2(self.dropout1(self.relu1(fy)))
        out_y = self.fc3(self.dropout2(self.relu2(fy)))
        return out_x, out_y, fx, fy

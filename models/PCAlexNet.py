# coding: utf-8

import torch
from torch import nn

from .pretrained import alexnet, alexnet_pre
from .BasicModule import BasicModule


class PCAlexNet(BasicModule):
    def __init__(self, num_classes):
        super(PCAlexNet, self).__init__()

        self.conv1 = alexnet.features[0:3]
        self.conv2 = alexnet.features[3:6]
        self.conv3 = alexnet.features[6:8]
        self.conv4 = alexnet.features[8:10]
        self.conv5 = alexnet.features[10:13]

        self.dropout1 = alexnet.classifier[0]
        self.fc1 = alexnet.classifier[1]
        self.relu1 = alexnet.classifier[2]
        self.dropout2 = alexnet.classifier[3]
        self.fc2 = alexnet.classifier[4]
        self.relu2 = alexnet.classifier[5]
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

    def forward(self, x, y):
        fx = self.conv1(x)
        fx = self.conv2(fx)
        fx = self.conv3(fx)
        fx = self.conv4(fx)
        fx = self.conv5(fx)
        fx = fx.view(fx.size(0), -1)

        fx = self.fc1(self.dropout1(fx))
        fx = self.fc2(self.dropout2(self.relu1(fx)))
        out_x = self.fc3(self.relu2(fx))

        fy = self.conv1(x)
        fy = self.conv2(fy)
        fy = self.conv3(fy)
        fy = self.conv4(fy)
        fy = self.conv5(fy)
        fy = fy.view(fy.size(0), -1)

        fy = self.fc1(self.dropout1(fy))
        fy = self.fc2(self.dropout2(self.relu1(fy)))
        out_y = self.fc3(self.relu2(fy))

        return out_x, out_y, fx, fy

# coding: utf-8

import torch
from torch import nn

from .pretrained import alexnet, alexnet_pre
from .BasicModule import BasicModule


class AlexNet(BasicModule):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

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

    def forward(self, x):
        features = self.conv1(x)
        features = self.conv2(features)
        features = self.conv3(features)
        features = self.conv4(features)
        features = self.conv5(features)
        features = features.view(features.size(0), -1)

        out = self.fc1(self.dropout1(features))
        out = self.fc2(self.dropout2(self.relu1(out)))
        out = self.fc3(self.relu2(out))

        return out


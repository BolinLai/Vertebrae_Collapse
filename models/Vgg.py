# coding: utf-8

import torch
from torch import nn

from .pretrained import vgg16, vgg16_pre
from .BasicModule import BasicModule


class Vgg16(BasicModule):
    def __init__(self, num_classes):
        super(Vgg16, self).__init__()

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

    def forward(self, x):
        features = self.conv1(x)
        features = self.conv2(features)
        features = self.conv3(features)
        features = self.conv4(features)
        features = self.conv5(features)
        features = features.view(features.size(0), -1)

        out = self.fc1(features)
        out = self.fc2(self.dropout1(self.relu1(out)))
        out = self.fc3(self.dropout2(self.relu2(out)))
        return out

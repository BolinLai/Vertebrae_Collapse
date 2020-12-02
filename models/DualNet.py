# coding: utf-8

import math
import torch
from torch import nn

from .pretrained import vgg16
from .BasicModule import BasicModule


class DualNet(BasicModule):
    def __init__(self, num_classes, init_weights=True):
        super(DualNet, self).__init__()
        # two channels
        # conv1 = [nn.Conv2d(2, 4, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(4, 4, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # conv2 = [nn.Conv2d(4, 8, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(8, 8, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # conv3 = [nn.Conv2d(8, 16, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(16, 16, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(16, 16, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # conv4 = [nn.Conv2d(16, 32, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(32, 32, kernel_size=3, padding=1),
        #          nn.ReLU(inplace=True),
        #          nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # self.features = nn.Sequential(*(conv1 + conv2 + conv3 + conv4))
        #
        # # self.classifier = nn.Linear(in_features=6272, out_features=num_classes)
        # self.classifier = nn.Linear(in_features=1568, out_features=num_classes)

        # two input branches
        conv1 = [nn.Conv2d(1, 4, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(4, 4, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(kernel_size=2, stride=2)]

        dual_conv1 = [nn.Conv2d(1, 4, kernel_size=3, padding=1),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(4, 4, kernel_size=3, padding=1),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(kernel_size=2, stride=2)]

        conv2 = [nn.Conv2d(4, 8, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(8, 8, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(kernel_size=2, stride=2)]

        dual_conv2 = [nn.Conv2d(4, 8, kernel_size=3, padding=1),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(8, 8, kernel_size=3, padding=1),
                      nn.ReLU(inplace=True),
                      nn.MaxPool2d(kernel_size=2, stride=2)]

        conv3 = [nn.Conv2d(16, 16, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(16, 16, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(16, 16, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(kernel_size=2, stride=2)]

        conv4 = [nn.Conv2d(16, 32, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(32, 32, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv2d(32, 32, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.MaxPool2d(kernel_size=2, stride=2)]

        self.ori_pip = nn.Sequential(*(conv1 + conv2))
        self.dual_pip = nn.Sequential(*(dual_conv1 + dual_conv2))
        self.concat_pip = nn.Sequential(*(conv3 + conv4))

        self.classifier = nn.Linear(in_features=1568, out_features=num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x, y):
        # two channels
        # features = self.features(torch.cat(x, y, 1))
        # features = features.view(features.size(0), -1)
        # out = self.classifier(features)

        # two input branches
        ori_features = self.ori_pip(x)
        dual_features = self.dual_pip(y)
        features = self.concat_pip(torch.cat((ori_features, dual_features), 1))
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    a = torch.rand(112, 112)
    b = torch.rand(112, 112)
    model = DualNet(num_classes=2)
    print(model(a, b))

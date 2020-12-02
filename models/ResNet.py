# coding: utf-8

import math
import copy
import torch
from torch import nn
from torch.nn import functional

from .pretrained import resnet18, resnet18_pre, resnet34, resnet34_pre, resnet50, resnet50_pre
from .BasicModule import BasicModule


class ResNet18(BasicModule):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()

        # train from scratch
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        self.avgpool = resnet18.avgpool
        # self.avgpool = nn.AvgPool2d(14, stride=1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # pre-trained
        # self.conv1 = resnet18_pre.conv1
        # self.bn1 = resnet18_pre.bn1
        # self.relu = resnet18_pre.relu
        # self.maxpool = resnet18_pre.maxpool
        # self.layer1 = resnet18_pre.layer1
        # self.layer2 = resnet18_pre.layer2
        # self.layer3 = resnet18_pre.layer3
        # self.layer4 = resnet18_pre.layer4
        # self.avgpool = resnet18_pre.avgpool
        # # self.avgpool = nn.AvgPool2d(14, stride=1)
        # self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        feature = self.conv1(x)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)

        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        feature = self.layer4(feature)

        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)
        return out

    def addfeatures(self):
        self.features = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                      self.layer1, self.layer2, self.layer3, self.layer4,
                                      self.avgpool)
        self.classifier = nn.Sequential(self.fc)


class SkipResNet18(BasicModule):
    def __init__(self, num_classes):
        super(SkipResNet18, self).__init__()

        # train from scratch
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4

        self.adapool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.adaconv = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1)

        self.avgpool = resnet18.avgpool
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        feature = self.conv1(x)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        skip_1 = self.adapool(feature)

        feature = self.layer1(feature)
        skip_2 = self.adapool(feature)

        feature = self.layer2(feature)
        skip_3 = self.adapool(feature)

        feature = self.layer3(feature)
        skip_4 = self.adapool(feature)

        feature = self.layer4(feature)

        feature = torch.cat((skip_1, skip_2, skip_3, skip_4, feature), 1)
        feature = self.adaconv(feature)

        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)
        return out


class DensResNet18(BasicModule):
    def __init__(self, num_classes):
        super(DensResNet18, self).__init__()

        # train from scratch
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4

        self.adapool56 = nn.AdaptiveAvgPool2d(output_size=(56, 56))
        self.adapool28 = nn.AdaptiveAvgPool2d(output_size=(28, 28))
        self.adapool14 = nn.AdaptiveAvgPool2d(output_size=(14, 14))
        self.adapool7 = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.adaconv1 = nn.Conv2d(in_channels=64+64, out_channels=64, kernel_size=1, stride=1)
        self.adaconv2 = nn.Conv2d(in_channels=64+64+128, out_channels=128, kernel_size=1, stride=1)
        self.adaconv3 = nn.Conv2d(in_channels=64+64+128+256, out_channels=256, kernel_size=1, stride=1)
        self.adaconv4 = nn.Conv2d(in_channels=64+64+128+256+512, out_channels=512, kernel_size=1, stride=1)

        self.avgpool = resnet18.avgpool
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        feature = self.conv1(x)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        f0 = self.maxpool(feature)

        f1 = self.layer1(f0)
        feature = torch.cat((self.adapool56(f0), f1), 1)
        feature = self.adaconv1(feature)

        f2 = self.layer2(feature)
        feature = torch.cat((self.adapool28(f0), self.adapool28(f1), f2), 1)
        feature = self.adaconv2(feature)

        f3 = self.layer3(feature)
        feature = torch.cat((self.adapool14(f0), self.adapool14(f1), self.adapool14(f2), f3), 1)
        feature = self.adaconv3(feature)

        f4 = self.layer4(feature)
        feature = torch.cat((self.adapool7(f0), self.adapool7(f1), self.adapool7(f2), self.adapool7(f3), f4), 1)
        feature = self.adaconv4(feature)

        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)
        return out


class GuideResNet18(BasicModule):
    def __init__(self, num_classes):
        super(GuideResNet18, self).__init__()

        # train from scratch
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4

        # self.upconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, stride=2, padding=1)
        # self.upconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.upconv1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1)
        self.upconv2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1)
        self.downconv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.downconv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)

        self.resblock1 = copy.deepcopy(resnet18.layer4)
        self.resblock2 = copy.deepcopy(resnet18.layer4)

        self.avgpool = resnet18.avgpool
        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        feature = self.conv1(x)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)

        feature = self.layer1(feature)
        feature = self.layer2(feature)
        f2 = feature
        feature = self.layer3(feature)
        f3 = feature
        feature = self.layer4(feature)

        up1 = self.upconv1(feature)
        up1 = functional.interpolate(up1, size=(14, 14), mode='bilinear', align_corners=False)
        down1 = self.downconv1(f3)
        up2 = self.upconv2(feature)
        up2 = functional.interpolate(up2, size=(14, 14), mode='bilinear', align_corners=False)
        down2 = self.downconv2(f2)

        guide_feature1 = self.resblock1(torch.cat((up1, down1), 1))
        guide_feature2 = self.resblock2(torch.cat((up2, down2), 1))

        feature = self.avgpool(torch.cat((guide_feature1, guide_feature2), 1))
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)
        return out


class ResNet34(BasicModule):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()

        # train from scratch
        self.conv1 = resnet34.conv1
        self.bn1 = resnet34.bn1
        self.relu = resnet34.relu
        self.maxpool = resnet34.maxpool
        self.layer1 = resnet34.layer1
        self.layer2 = resnet34.layer2
        self.layer3 = resnet34.layer3
        self.layer4 = resnet34.layer4
        self.avgpool = resnet34.avgpool
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        feature = self.conv1(x)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)

        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        feature = self.layer4(feature)

        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)
        return out


class ResNet50(BasicModule):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()

        # train from scratch
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

    def forward(self, x):
        feature = self.conv1(x)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)

        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        feature = self.layer4(feature)

        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)
        return out

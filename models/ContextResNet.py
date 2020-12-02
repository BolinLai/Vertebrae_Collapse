# coding: utf-8

import math
import copy
import torch
from torch import nn
from torch.nn import functional

from .pretrained import resnet18, resnet18_pre, resnet34, resnet34_pre, resnet50, resnet50_pre
from .BasicModule import BasicModule


class ContextResNet18(BasicModule):
    def __init__(self, num_classes):
        super(ContextResNet18, self).__init__()

        # train from scratch
        self.conv1_1 = resnet18.conv1
        self.bn1_1 = resnet18.bn1

        self.conv1_2 = copy.deepcopy(resnet18.conv1)
        self.bn1_2 = copy.deepcopy(resnet18.bn1)

        self.conv1_3 = copy.deepcopy(resnet18.conv1)
        self.bn1_3 = copy.deepcopy(resnet18.bn1)

        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool

        self.layer1_1 = resnet18.layer1
        self.layer1_2 = copy.deepcopy(resnet18.layer1)
        self.layer1_3 = copy.deepcopy(resnet18.layer1)

        self.layer2_1 = resnet18.layer2
        self.layer2_2 = copy.deepcopy(resnet18.layer2)
        self.layer2_3 = copy.deepcopy(resnet18.layer2)

        # for concat
        # self.conv = nn.Conv2d(128 * 3, 128, kernel_size=3, stride=1, padding=1)
        # self.bn = nn.BatchNorm2d(128)
        # self.relu2 = nn.ReLU(inplace=True)

        # for two MSE loss
        self.dconv1 = nn.Conv2d(128 * 2, 256, kernel_size=3, stride=2, padding=1)
        self.drelu1 = nn.ReLU(inplace=True)
        self.dconv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.drelu2 = nn.ReLU(inplace=True)
        self.dpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dfc = nn.Linear(256, 1)

        # for two MSE loss of logits
        # self.dfc1 = nn.Linear(1024, 1)
        # self.dfc2 = nn.Linear(1024, 1)

        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4

        # self.layer3_1 = resnet18.layer3
        # self.layer3_2 = copy.deepcopy(resnet18.layer3)
        # self.layer3_3 = copy.deepcopy(resnet18.layer3)
        # self.layer4_1 = resnet18.layer4
        # self.layer4_2 = copy.deepcopy(resnet18.layer4)
        # self.layer4_3 = copy.deepcopy(resnet18.layer4)

        self.avgpool = resnet18.avgpool
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2, x3):
        # 在feature层面进行操作
        f1 = self.conv1_1(x1)
        f1 = self.bn1_1(f1)
        f1 = self.relu(f1)
        f1 = self.maxpool(f1)
        f1 = self.layer1_1(f1)
        f1 = self.layer2_1(f1)

        f2 = self.conv1_2(x2)
        f2 = self.bn1_2(f2)
        f2 = self.relu(f2)
        f2 = self.maxpool(f2)
        f2 = self.layer1_2(f2)
        f2 = self.layer2_2(f2)

        f3 = self.conv1_3(x3)
        f3 = self.bn1_3(f3)
        f3 = self.relu(f3)
        f3 = self.maxpool(f3)
        f3 = self.layer1_3(f3)
        f3 = self.layer2_3(f3)

        # concat
        # feature = torch.cat((f1, f2, f3), 1)
        # feature = self.conv(feature)
        # feature = self.bn(feature)
        # feature = self.relu2(feature)

        # add in feature
        feature = f1 + f2 + f3

        # two MSE loss
        df1_2 = self.drelu2(self.dconv2(self.drelu1(self.dconv1(torch.cat((f1, f2), 1)))))
        df2_3 = self.drelu2(self.dconv2(self.drelu1(self.dconv1(torch.cat((f2, f3), 1)))))
        diff1 = self.dfc(self.dpool(df1_2).view(df1_2.size(0), -1))
        diff2 = self.dfc(self.dpool(df2_3).view(df2_3.size(0), -1))

        feature = self.layer3(feature)
        feature = self.layer4(feature)

        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)

        # 对logits进行操作
        # f1 = self.conv1_1(x1)
        # f1 = self.bn1_1(f1)
        # f1 = self.relu(f1)
        # f1 = self.maxpool(f1)
        # f1 = self.layer1_1(f1)
        # f1 = self.layer2_1(f1)
        # f1 = self.layer3_1(f1)
        # f1 = self.layer4_1(f1)
        # f1 = self.avgpool(f1)
        #
        # f2 = self.conv1_2(x2)
        # f2 = self.bn1_2(f2)
        # f2 = self.relu(f2)
        # f2 = self.maxpool(f2)
        # f2 = self.layer1_2(f2)
        # f2 = self.layer2_2(f2)
        # f2 = self.layer3_2(f2)
        # f2 = self.layer4_2(f2)
        # f2 = self.avgpool(f2)
        #
        # f3 = self.conv1_3(x3)
        # f3 = self.bn1_3(f3)
        # f3 = self.relu(f3)
        # f3 = self.maxpool(f3)
        # f3 = self.layer1_3(f3)
        # f3 = self.layer2_3(f3)
        # f3 = self.layer3_3(f3)
        # f3 = self.layer4_3(f3)
        # f3 = self.avgpool(f3)
        #
        # # two MSE loss
        # diff1 = self.dfc1(torch.cat((f1, f2), 1).view(f2.size(0), -1))
        # diff2 = self.dfc2(torch.cat((f2, f3), 1).view(f2.size(0), -1))
        #
        # feature = f1 + f2 + f3
        # feature = feature.view(feature.size(0), -1)
        # out = self.fc(feature)

        # return out
        return out, diff1, diff2

        # for L2 norm
        # return out, f1, f2, f3


class ContextShareNet(BasicModule):
    def __init__(self, num_classes):
        super(ContextShareNet, self).__init__()

        # train from scratch
        self.conv1_1 = resnet18.conv1
        self.bn1_1 = resnet18.bn1

        self.conv1_2 = resnet18.conv1
        self.bn1_2 = resnet18.bn1

        self.conv1_3 = resnet18.conv1
        self.bn1_3 = resnet18.bn1

        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool

        self.layer1_1 = resnet18.layer1
        self.layer1_2 = resnet18.layer1
        self.layer1_3 = resnet18.layer1

        self.layer2_1 = resnet18.layer2
        self.layer2_2 = resnet18.layer2
        self.layer2_3 = resnet18.layer2

        # for concat
        # self.conv = nn.Conv2d(128 * 3, 128, kernel_size=3, stride=1, padding=1)
        # self.bn = nn.BatchNorm2d(128)
        # self.relu2 = nn.ReLU(inplace=True)

        # for two MSE loss
        # self.dconv1 = nn.Conv2d(128 * 2, 256, kernel_size=3, stride=2, padding=1)
        # self.drelu1 = nn.ReLU(inplace=True)
        # self.dconv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        # self.drelu2 = nn.ReLU(inplace=True)
        # self.dpool = nn.AvgPool2d(kernel_size=7, stride=1)
        # self.dfc = nn.Linear(256, 1)

        # for two MSE loss of logits
        # self.dfc1 = nn.Linear(1024, 1)
        # self.dfc2 = nn.Linear(1024, 1)

        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4

        # self.layer3_1 = copy.deepcopy(resnet18.layer3)
        # self.layer3_3 = copy.deepcopy(resnet18.layer3)
        # self.layer4_1 = copy.deepcopy(resnet18.layer4)
        # self.layer4_3 = copy.deepcopy(resnet18.layer4)

        self.avgpool = resnet18.avgpool
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2, x3):
        # 在feature层面进行操作
        f1 = self.conv1_1(x1)
        f1 = self.bn1_1(f1)
        f1 = self.relu(f1)
        f1 = self.maxpool(f1)
        f1 = self.layer1_1(f1)
        f1 = self.layer2_1(f1)

        f2 = self.conv1_2(x2)
        f2 = self.bn1_2(f2)
        f2 = self.relu(f2)
        f2 = self.maxpool(f2)
        f2 = self.layer1_2(f2)
        f2 = self.layer2_2(f2)

        f3 = self.conv1_3(x3)
        f3 = self.bn1_3(f3)
        f3 = self.relu(f3)
        f3 = self.maxpool(f3)
        f3 = self.layer1_3(f3)
        f3 = self.layer2_3(f3)

        # concat
        # feature = torch.cat((f1, f2, f3), 1)

        # feature = self.conv(feature)
        # feature = self.bn(feature)
        # feature = self.relu2(feature)

        # add in feature
        feature = f1 + f2 + f3

        # two MSE loss
        # df1_2 = self.drelu2(self.dconv2(self.drelu1(self.dconv1(torch.cat((f1, f2), 1)))))
        # df2_3 = self.drelu2(self.dconv2(self.drelu1(self.dconv1(torch.cat((f2, f3), 1)))))
        # diff1 = self.dfc(self.dpool(df1_2).view(df1_2.size(0), -1))
        # diff2 = self.dfc(self.dpool(df2_3).view(df2_3.size(0), -1))

        feature = self.layer3(feature)
        feature = self.layer4(feature)

        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)

        # 对logits进行操作
        # f1 = self.conv1_1(x1)
        # f1 = self.bn1_1(f1)
        # f1 = self.relu(f1)
        # f1 = self.maxpool(f1)
        # f1 = self.layer1_1(f1)
        # f1 = self.layer2_1(f1)
        # f1 = self.layer3_1(f1)
        # f1 = self.layer4_1(f1)
        # f1 = self.avgpool(f1)
        #
        # f2 = self.conv1_2(x2)
        # f2 = self.bn1_2(f2)
        # f2 = self.relu(f2)
        # f2 = self.maxpool(f2)
        # f2 = self.layer1_2(f2)
        # f2 = self.layer2_2(f2)
        # f2 = self.layer3(f2)
        # f2 = self.layer4(f2)
        # f2 = self.avgpool(f2)
        #
        # f3 = self.conv1_3(x3)
        # f3 = self.bn1_3(f3)
        # f3 = self.relu(f3)
        # f3 = self.maxpool(f3)
        # f3 = self.layer1_3(f3)
        # f3 = self.layer2_3(f3)
        # f3 = self.layer3_3(f3)
        # f3 = self.layer4_3(f3)
        # f3 = self.avgpool(f3)
        #
        # # two MSE loss
        # diff1 = self.dfc1(torch.cat((f1, f2), 1).view(f2.size(0), -1))
        # diff2 = self.dfc2(torch.cat((f2, f3), 1).view(f2.size(0), -1))
        #
        # feature = f1 + f2 + f3
        # feature = feature.view(feature.size(0), -1)
        # out = self.fc(feature)

        # return out
        # return out, diff1, diff2

        # for L2 norm
        return out, f1, f2, f3


class ContextResNet50(BasicModule):
    def __init__(self, num_classes):
        super(ContextResNet50, self).__init__()

        # train from scratch
        self.conv1_1 = resnet50.conv1
        self.bn1_1 = resnet50.bn1

        self.conv1_2 = copy.deepcopy(resnet50.conv1)
        self.bn1_2 = copy.deepcopy(resnet50.bn1)

        self.conv1_3 = copy.deepcopy(resnet50.conv1)
        self.bn1_3 = copy.deepcopy(resnet50.bn1)

        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool

        self.layer1_1 = resnet50.layer1
        self.layer1_2 = copy.deepcopy(resnet50.layer1)
        self.layer1_3 = copy.deepcopy(resnet50.layer1)

        self.layer2_1 = resnet50.layer2
        self.layer2_2 = copy.deepcopy(resnet50.layer2)
        self.layer2_3 = copy.deepcopy(resnet50.layer2)

        # for two MSE loss
        self.dconv1 = nn.Conv2d(512 * 2, 1024, kernel_size=3, stride=2, padding=1)
        self.drelu1 = nn.ReLU(inplace=True)
        self.dconv2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)
        self.drelu2 = nn.ReLU(inplace=True)
        self.dpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dfc = nn.Linear(1024, 1)

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

    def forward(self, x1, x2, x3):
        f1 = self.conv1_1(x1)
        f1 = self.bn1_1(f1)
        f1 = self.relu(f1)
        f1 = self.maxpool(f1)
        f1 = self.layer1_1(f1)
        f1 = self.layer2_1(f1)

        f2 = self.conv1_2(x2)
        f2 = self.bn1_2(f2)
        f2 = self.relu(f2)
        f2 = self.maxpool(f2)
        f2 = self.layer1_2(f2)
        f2 = self.layer2_2(f2)

        f3 = self.conv1_3(x3)
        f3 = self.bn1_3(f3)
        f3 = self.relu(f3)
        f3 = self.maxpool(f3)
        f3 = self.layer1_3(f3)
        f3 = self.layer2_3(f3)

        # add in feature
        feature = f1 + f2 + f3

        # two MSE loss
        df1_2 = self.drelu2(self.dconv2(self.drelu1(self.dconv1(torch.cat((f1, f2), 1)))))
        df2_3 = self.drelu2(self.dconv2(self.drelu1(self.dconv1(torch.cat((f2, f3), 1)))))
        diff1 = self.dfc(self.dpool(df1_2).view(df1_2.size(0), -1))
        diff2 = self.dfc(self.dpool(df2_3).view(df2_3.size(0), -1))

        feature = self.layer3(feature)
        feature = self.layer4(feature)

        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        out = self.fc(feature)

        return out, diff1, diff2

# coding: utf-8

import torch
from torch import nn
from torch.nn import functional
from torchvision import models

from .BasicModule import BasicModule

alexnet_pre = models.alexnet(pretrained=True)
# alexnet_pre.cuda()

alexnet = models.alexnet(pretrained=False)
# alexnet.cuda()

vgg16_pre = models.vgg16(pretrained=True)
# vgg16_pre.cuda()

vgg16 = models.vgg16(pretrained=False)
# vgg16.cuda()

resnet18_pre = models.resnet18(pretrained=True)
# resnet18_pre.cuda()

resnet18 = models.resnet18(pretrained=False)
# resnet18.cuda()

resnet34_pre = models.resnet34(pretrained=True)
# resnet34_pre.cuda()

resnet34 = models.resnet34(pretrained=False)
# resnet34.cuda()

resnet50_pre = models.resnet50(pretrained=True)
# resnet50_pre.cuda()

resnet50 = models.resnet50(pretrained=False)
# resnet50.cuda()

densenet121_pre = models.densenet121(pretrained=True)
# densenet121_pre.cuda()

densenet121 = models.densenet121(pretrained=False)
# densenet121.cuda()

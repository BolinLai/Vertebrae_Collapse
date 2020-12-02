# coding = utf-8

import torch

from models import PCResNet18
from torch.utils.data import DataLoader

from dataset import ContextVB_Dataset


# model = PCResNet18(num_classes=3)
# a = torch.rand(32, 3, 224, 224)
# b = torch.rand(32, 3, 224, 224)
# score1, score2, logits1, logits2 = model(a, b)
# print(score1.size(), score2.size(), logits1.size(), logits2.size())

train_data = ContextVB_Dataset(csv_path=['dataset/train_ygy.csv', 'dataset/val_ygy.csv'], num_classes=2, phase='train', useRGB=True, usetrans=True, balance='upsample')
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)

for _, _, _ in train_dataloader:
    pass

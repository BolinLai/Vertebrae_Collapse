# coding: utf-8

import os
import random
import numpy as np
import torch
import csv

from torchvision import transforms
from torchvision.transforms import functional
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm


class CollapseDataset(object):
    def __init__(self, csv_path, phase, useRGB=True, usetrans=True, balance=False):

        self.csv_path = csv_path
        self.phase = phase
        self.useRGB = useRGB
        self.usetrans = usetrans
        self.balance = balance

        self.images, self.labels = self.prepare_data()

        if self.usetrans:
            if self.phase == 'train':
                self.trans = transforms.Compose([
                    # transforms.Resize((224, 224)),
                    # transforms.Resize((448, 448)),
                    transforms.Resize((512, 512)),
                    transforms.RandomCrop((448, 448)),
                    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(30),
                    transforms.ToTensor(),
                ])
            elif self.phase == 'val' or self.phase == 'test' or self.phase == 'test_train':
                self.trans = transforms.Compose([
                    # transforms.Resize((224, 224)),
                    transforms.Resize((448, 448)),
                    transforms.ToTensor(),
                ])
            else:
                raise IndexError
        else:
            self.trans = transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path)
        image = Image.fromarray(np.asarray(image)[:, :, 0]) if not self.useRGB else image  # 得到的RGB图片三通道数值相等，只选择其中一个
        image = self.trans(image)

        label = self.labels[index]

        return image, label, image_path

    def prepare_data(self):
        lines = []
        for csv_file in self.csv_path:
            with open(csv_file, 'r') as f:
                lines.extend(f.readlines())
            f.close()
        if self.phase == 'train' or self.phase == 'val':
            random.shuffle(lines)
        elif self.phase == 'test' or self.phase == 'test_train':
            pass
            # lines.sort(key=lambda x: (x.split(',')[0].split('/')[-2], int(x.split(',')[0].split('/')[-1].split('_')[0][2:])))
        else:
            raise ValueError

        images = [str(x).strip().split(',')[0] for x in tqdm(lines, desc='Preparing Images')]
        labels = [int(str(x).strip().split(',')[1]) for x in tqdm(lines, desc='Preparing Labels')]

        return images, labels

    def dist(self):
        dist = {}
        for l in tqdm(self.labels, desc="Counting data distribution"):
            if str(l) in dist.keys():
                dist[str(l)] += 1
            else:
                dist[str(l)] = 1
        return dist


if __name__ == '__main__':
    train_data = CollapseDataset(csv_path=['dataset/train_single.csv'], phase='train', useRGB=True, usetrans=True, balance=False)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)

    for img, lab, img_path in tqdm(train_dataloader):
        # print(img.size())
        # tqdm.write(f'{img.mean()}, {img.std()}')
        # raise KeyboardInterrupt
        pass

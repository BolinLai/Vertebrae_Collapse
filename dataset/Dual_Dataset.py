# coding: utf-8

import os
import csv
import random
import torch
import PIL
import numpy as np

from torchvision import transforms
from torchvision.transforms import functional
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm


class Dual_Dataset(object):
    def __init__(self, csv_path, phase, useRGB=True, usetrans=True, padding=False, balance=False):

        self.csv_path = csv_path
        self.phase = phase
        self.useRGB = useRGB
        self.usetrans = usetrans
        self.balance = balance
        self.padding = padding

        self.images, self.labels = self.prepare_data()

        if self.usetrans:
            if self.phase == 'train':
                self.trans = transforms.Compose([
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomVerticalFlip(),
                    # transforms.RandomRotation(30),
                    transforms.ToTensor(),
                ])

            elif self.phase == 'val' or self.phase == 'test' or self.phase == 'test_train':
                self.trans = transforms.Compose([
                    transforms.ToTensor(),
                ])

            else:
                raise IndexError
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        dual_path = image_path.replace('SW_VB', 'VB_TruncDoG_1.0&0.5&0.6')

        image = Image.open(image_path)
        image = Image.fromarray(np.asarray(image)[:, :, 0]) if not self.useRGB else image  # 得到的RGB图片三通道数值相等，只选择其中一个
        dual_image = Image.open(dual_path)
        dual_image = Image.fromarray(np.asarray(dual_image)[:, :, 0]) if not self.useRGB else image  # 得到的RGB图片三通道数值相等，只选择其中一个

        if self.padding:  # 调整图像长边为112，以下代码出自torchvision.transforms.functional.resize
            size = 112
            w, h = image.size
            if max(w, h) == size:
                ow, oh = w, h
                pass
            elif w < h:
                ow = int(size * w / h)
                oh = size
                image = image.resize((ow, oh), resample=Image.BILINEAR)
            else:
                ow = size
                oh = int(size * h / w)
                image = image.resize((ow, oh), resample=Image.BILINEAR)

            # 将短边补齐到112
            image = functional.pad(image, fill=0, padding_mode='constant',
                                   padding=((size - ow) // 2, (size - oh) // 2,
                                            (size - ow) - (size - ow) // 2, (size - oh) - (size - oh) // 2))
        else:  # resize到112*112
            image = functional.resize(image, (112, 112))
            dual_image = functional.resize(dual_image, (112, 112))

        if self.usetrans:
            if random.random() < 0.5:
                image = functional.hflip(image)
                dual_image = functional.hflip(dual_image)
            if random.random() < 0.5:
                image = functional.vflip(image)
                dual_image = functional.vflip(dual_image)
            angle = random.uniform(-30, 30)
            image = functional.rotate(image, angle, resample=False, expand=False, center=None)
            dual_image = functional.rotate(dual_image, angle, resample=PIL.Image.BILINEAR, expand=False, center=None)

        image = self.trans(image)
        dual_image = self.trans(dual_image)

        label = self.labels[index]

        return image, dual_image, label, image_path

    def prepare_data(self):
        lines = []
        for csv_file in self.csv_path:
            with open(csv_file, 'r') as f:
                lines.extend(f.readlines())
            f.close()
        if self.phase == 'train':
            if self.balance:
                # 2分类
                positive = [line for line in lines if int(str(line).strip().split(',')[1]) in [2, 3]]
                negative = [line for line in lines if int(str(line).strip().split(',')[1]) in [0, 1]]
                if self.balance == 'upsample':
                    positive = positive * (len(negative) // len(positive))  # 增加正样本
                elif self.balance == 'downsample':
                    negative = random.sample(negative, len(positive))  # 减少负样本
                else:
                    raise ValueError
                lines = positive + negative

                # 3分类
                # negative = [line for line in lines if int(str(line).strip().split(',')[1]) in [0, 1]]
                # collapse1 = [line for line in lines if int(str(line).strip().split(',')[1]) == 2]
                # collapse2 = [line for line in lines if int(str(line).strip().split(',')[1]) == 3]
                # if self.balance == 'upsample':
                #     collapse1 = collapse1 * (len(negative) // len(collapse1))
                #     collapse2 = collapse2 * (len(negative) // len(collapse2))
                # else:
                #     raise ValueError
                # lines = negative + collapse1 + collapse2

            random.shuffle(lines)
        elif self.phase == 'val':
            random.shuffle(lines)
        elif self.phase == 'test' or self.phase == 'test_train':
            pass
        else:
            raise ValueError

        images = [str(x).strip().split(',')[0] for x in tqdm(lines, desc='Preparing Images')]
        # 2分类
        labels = [0 if int(str(x).strip().split(',')[1]) in [0, 1] else 1 for x in tqdm(lines, desc='Preparing Labels')]
        # 3分类
        # labels = [0 if int(str(x).strip().split(',')[1]) in [0, 1] else int(str(x).strip().split(',')[1])-1 for x in tqdm(lines, desc='Preparing Labels')]

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
    train_data = Dual_Dataset(csv_path=['dataset/train_VB.csv', 'dataset/val_VB.csv'], phase='train', useRGB=False, usetrans=True, balance='upsample')
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)

    count = 0
    for img, dimg, lab, img_path in tqdm(train_data):
        tqdm.write(f"{count}: {img_path}")
        img = Image.fromarray(np.uint8(img.squeeze().numpy() * 255))  # img尺寸(1, 112, 112)，Image转换需要尺寸(112, 112) / (112, 112, 1)
        img.save(f'/DB/rhome/bllai/Data/DATA3/Vertebrae/Sagittal/test/{count}_{lab}.png')
        dimg = Image.fromarray(np.uint8(dimg.squeeze().numpy() * 255))  # img尺寸(1, 112, 112)，Image转换需要尺寸(112, 112) / (112, 112, 1)
        dimg.save(f'/DB/rhome/bllai/Data/DATA3/Vertebrae/Sagittal/test/{count}_{lab}d.png')

        count += 1
        if count == 20:
            raise KeyboardInterrupt

        # tqdm.write(str(img.size))
        # assert img.size == (224, 224) and lab in [0, 1]
        # tqdm.write(f'{img.mean()}, {img.std()}')
        # pass

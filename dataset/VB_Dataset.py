# coding: utf-8

import os
import csv
import random
import torch
import numpy as np

from torchvision import transforms
from torchvision.transforms import functional
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm


class VB_Dataset(object):
    def __init__(self, csv_path, phase, num_classes, useRGB=True, usetrans=True, padding=False, balance=False):

        self.csv_path = csv_path
        self.phase = phase
        self.num_classes = num_classes
        self.useRGB = useRGB
        self.usetrans = usetrans
        self.balance = balance
        self.padding = padding
        self.scale = []

        self.images, self.labels = self.prepare_data()

        if self.usetrans:
            if self.phase == 'train':
                self.trans = transforms.Compose([
                    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(30),
                    transforms.ToTensor(),
                ])

                # self.neg_trans = transforms.Compose([
                #     transforms.RandomHorizontalFlip(),
                #     transforms.RandomVerticalFlip(),
                #     transforms.RandomRotation(30),
                #     transforms.ToTensor(),
                # ])
            elif self.phase == 'val' or self.phase == 'test' or self.phase == 'test_train':
                self.trans = transforms.Compose([
                    transforms.ToTensor(),
                ])

                # self.neg_trans = self.trans
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
        # image_path = image_path.replace('SW_VBCus', 'SW_VBSoft')
        # image_path = image_path.replace('/DB/rhome/bllai/Data/DATA3/Vertebrae/Sagittal',   # for ai-research server
        #                                 '/mnt/lustre/ai-vision/home/yz891/bllai/Data/Vertebrae_Collapse')
        image = Image.open(image_path)
        image = Image.fromarray(np.asarray(image)[:, :, 0]) if not self.useRGB else image  # 得到的RGB图片三通道数值相等，只选择其中一个

        if self.padding:  # 调整图像长边为224，以下代码出自torchvision.transforms.functional.resize
            size = 224
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

            # 将短边补齐到224
            image = functional.pad(image, fill=0, padding_mode='constant',
                                   padding=((size - ow) // 2, (size - oh) // 2,
                                            (size - ow) - (size - ow) // 2, (size - oh) - (size - oh) // 2))
        else:  # resize到224*224
            image = functional.resize(image, (224, 224))

        image = self.trans(image)

        label = self.labels[index]

        # if label == 0:
        #     image = self.neg_trans(image)
        # else:
        #     image = self.trans(image)

        return image, label, image_path

    def prepare_data(self):
        lines = []
        for csv_file in self.csv_path:
            with open(csv_file, 'r') as f:
                lines.extend(f.readlines())
            f.close()

        if self.balance:
            # 2分类
            if self.num_classes == 2:
                positive = [line for line in lines if int(str(line).strip().split(',')[1]) in [2, 3]]
                negative = [line for line in lines if int(str(line).strip().split(',')[1]) in [0, 1]]
                if self.balance == 'upsample':
                    self.scale = [1, len(negative) // len(positive)]
                    positive = positive * (len(negative) // len(positive))  # 增加正样本
                elif self.balance == 'downsample':
                    negative = random.sample(negative, 2*len(positive))  # 减少负样本
                    self.scale = [len(negative) // len(positive), 1]
                else:
                    raise ValueError
                lines = positive + negative

            # 3分类
            elif self.num_classes == 3:
                negative = [line for line in lines if int(str(line).strip().split(',')[1]) in [0, 1]]
                collapse1 = [line for line in lines if int(str(line).strip().split(',')[1]) == 2]
                collapse2 = [line for line in lines if int(str(line).strip().split(',')[1]) == 3]
                if self.balance == 'upsample':
                    self.scale = [1, len(negative) // len(collapse1), len(negative) // len(collapse2)]
                    collapse1 = collapse1 * (len(negative) // len(collapse1))
                    collapse2 = collapse2 * (len(negative) // len(collapse2))
                elif self.balance == 'tSNE':  # only for tSNE
                    self.scale = [1, 1, len(collapse1) // len(collapse2)]
                    with open('dataset/tSNE_idx2.csv', 'r') as f:
                        sample = f.readlines()
                    f.close()
                    negative = [negative[int(item)] for item in sample]
                    collapse2 = collapse2 * (len(collapse1) // len(collapse2))
                else:
                    raise ValueError
                lines = negative + collapse1 + collapse2
            else:
                raise ValueError

        if self.phase == 'train' or self.phase == 'val':
            random.shuffle(lines)
        elif self.phase == 'test' or self.phase == 'test_train':
            pass
        else:
            raise ValueError

        images = [str(x).strip().split(',')[0] for x in tqdm(lines, desc='Preparing Images')]
        # 2分类
        if self.num_classes == 2:
            labels = [0 if int(str(x).strip().split(',')[1]) in [0, 1] else 1 for x in tqdm(lines, desc='Preparing Labels')]
        # 3分类
        elif self.num_classes == 3:
            labels = [0 if int(str(x).strip().split(',')[1]) in [0, 1] else int(str(x).strip().split(',')[1])-1 for x in tqdm(lines, desc='Preparing Labels')]
        else:
            raise ValueError

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
    train_data = VB_Dataset(csv_path=['dataset/train_VB.csv', 'dataset/val_VB.csv'], phase='train', useRGB=False, usetrans=True, balance='upsample')
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)

    count = 0
    for img, lab, img_path in tqdm(train_data):
        img = Image.fromarray(np.uint8(img.squeeze().numpy() * 255))  # img尺寸(1, 112, 112)，Image转换需要尺寸(112, 112) / (112, 112, 1)
        img.save(f'/DB/rhome/bllai/Data/DATA3/Vertebrae/Sagittal/test/{count}_{lab}.png')
        count += 1
        if count == 20:
            raise KeyboardInterrupt

        # tqdm.write(str(img.size))
        # assert img.size == (224, 224) and lab in [0, 1]
        # tqdm.write(f'{img.mean()}, {img.std()}')
        pass

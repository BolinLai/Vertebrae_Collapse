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
from utils import write_csv


class ContextVB_Dataset(object):
    def __init__(self, csv_path, phase, num_classes, useRGB=True, usetrans=True, padding=False, balance='upsample'):

        self.csv_path = csv_path
        self.phase = phase
        self.num_classes = num_classes
        self.useRGB = useRGB
        self.usetrans = usetrans
        self.balance = balance
        self.padding = padding
        self.scale = []

        self.images, self.labels = self.prepare_data()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        last_image_path, image_path, next_image_path = self.images[index]
        # last_image_path = last_image_path.replace('/DB/rhome/bllai/Data/DATA3/Vertebrae/Sagittal',  # for ai-research server
        #                                           '/mnt/lustre/ai-vision/home/yz891/bllai/Data/Vertebrae_Collapse')
        # image_path = image_path.replace('/DB/rhome/bllai/Data/DATA3/Vertebrae/Sagittal',  # for ai-research server
        #                                 '/mnt/lustre/ai-vision/home/yz891/bllai/Data/Vertebrae_Collapse')
        # next_image_path = next_image_path.replace('/DB/rhome/bllai/Data/DATA3/Vertebrae/Sagittal',  # for ai-research server
        #                                           '/mnt/lustre/ai-vision/home/yz891/bllai/Data/Vertebrae_Collapse')

        last_image = Image.open(last_image_path)
        image = Image.open(image_path)
        next_image = Image.open(next_image_path)

        last_image = Image.fromarray(np.asarray(last_image)[:, :, 0]) if not self.useRGB else last_image
        image = Image.fromarray(np.asarray(image)[:, :, 0]) if not self.useRGB else image  # 得到的RGB图片三通道数值相等，只选择其中一个
        next_image = Image.fromarray(np.asarray(next_image)[:, :, 0]) if not self.useRGB else next_image

        if self.padding:  # 调整图像长边为224，以下代码出自torchvision.transforms.functional.resize
            size = 224
            w, h = image.size
            if max(w, h) == size:
                ow, oh = w, h
                pass
            elif w < h:
                ow = int(size * w / h)
                oh = size
                last_image = last_image.resize((ow, oh), resample=Image.BILINEAR)
                image = image.resize((ow, oh), resample=Image.BILINEAR)
                next_image = next_image.resize((ow, oh), resample=Image.BILINEAR)
            else:
                ow = size
                oh = int(size * h / w)
                last_image = last_image.resize((ow, oh), resample=Image.BILINEAR)
                image = image.resize((ow, oh), resample=Image.BILINEAR)
                next_image = next_image.resize((ow, oh), resample=Image.BILINEAR)

            # 将短边补齐到224
            last_image = functional.pad(last_image, fill=0, padding_mode='constant',
                                        padding=((size - ow) // 2, (size - oh) // 2,
                                                 (size - ow) - (size - ow) // 2, (size - oh) - (size - oh) // 2))

            image = functional.pad(image, fill=0, padding_mode='constant',
                                   padding=((size - ow) // 2, (size - oh) // 2,
                                            (size - ow) - (size - ow) // 2, (size - oh) - (size - oh) // 2))

            next_image = functional.pad(next_image, fill=0, padding_mode='constant',
                                        padding=((size - ow) // 2, (size - oh) // 2,
                                                 (size - ow) - (size - ow) // 2, (size - oh) - (size - oh) // 2))
        else:  # resize到224*224
            last_image = functional.resize(last_image, (224, 224))
            image = functional.resize(image, (224, 224))
            next_image = functional.resize(next_image, (224, 224))

        # 三块脊骨做一样的transformation
        if self.usetrans:
            # Random horizontal flip
            if random.random() < 0.5:
                last_image = functional.hflip(last_image)
                image = functional.hflip(image)
                next_image = functional.hflip(next_image)

            # Random vertical flip
            if random.random() < 0.5:
                last_image = functional.vflip(last_image)
                image = functional.vflip(image)
                next_image = functional.vflip(next_image)

            # Random rotation
            angle = random.uniform(-30, 30)
            last_image = functional.rotate(last_image, angle)
            image = functional.rotate(image, angle)
            next_image = functional.rotate(next_image, angle)

        # Convert to Tensor
        last_image = functional.to_tensor(last_image)
        image = functional.to_tensor(image)
        next_image = functional.to_tensor(next_image)

        # 三块脊骨分别做transformation
        # if self.trans:
        #     last_image = self.trans(last_image)
        #     image = self.trans(image)
        #     next_image = self.trans(next_image)

        last_label, label, next_label = self.labels[index]

        return (last_image, image, next_image), (last_label, label, next_label), (last_image_path, image_path, next_image_path)

    def prepare_data(self):
        lines = []
        for csv_file in self.csv_path:
            with open(csv_file, 'r') as f:
                lines.extend(f.readlines())
            f.close()

        if self.balance:
            # 2分类
            if self.num_classes == 2:
                negative, positive = [], []

                for i, line in enumerate(lines):
                    path, label = str(line).strip().split(',')

                    # 如果是每个病人的第一个patch，将本身作为上一块脊骨
                    if i == 0 or path.split('/')[10] != str(lines[i - 1]).strip().split(',')[0].split('/')[10]:
                        context = [line, line]
                    else:
                        context = [lines[i - 1], line]

                    # 如果是每个病人的最后一个patch，将本身作为下一块脊骨
                    if i == len(lines) - 1 or path.split('/')[10] != str(lines[i + 1]).strip().split(',')[0].split('/')[10]:
                        context.append(line)
                    else:
                        context.append(lines[i + 1])

                    # negative，positive中每个元素都是三元组，分别为相邻三块脊骨
                    if int(label) in [0, 1]:
                        negative.append(context)
                    elif int(label) in [2, 3]:
                        positive.append(context)
                    else:
                        raise ValueError

                if self.balance == 'upsample':
                    self.scale = [1, len(negative) // len(positive)]
                    positive = positive * (len(negative) // len(positive))
                else:
                    raise ValueError

                lines = positive + negative

            # 3分类
            elif self.num_classes == 3:
                negative, collapse1, collapse2 = [], [], []

                for i, line in enumerate(lines):
                    path, label = str(line).strip().split(',')

                    # 如果是每个病人的第一个patch，将本身作为上一块脊骨
                    if i == 0 or path.split('/')[10] != str(lines[i-1]).strip().split(',')[0].split('/')[10]:
                        context = [line, line]
                    else:
                        context = [lines[i-1], line]

                    # 如果是每个病人的最后一个patch，将本身作为下一块脊骨
                    if i == len(lines)-1 or path.split('/')[10] != str(lines[i+1]).strip().split(',')[0].split('/')[10]:
                        context.append(line)
                    else:
                        context.append(lines[i+1])

                    # negative，collapse1，collapse2中每个元素都是三元组，分别为相邻三块脊骨
                    if int(label) in [0, 1]:
                        negative.append(context)
                    elif int(label) == 2:
                        collapse1.append(context)
                    elif int(label) == 3:
                        collapse2.append(context)
                    else:
                        raise ValueError

                if self.balance == 'upsample':
                    self.scale = [1, len(negative) // len(collapse1), len(negative) // len(collapse2)]
                    collapse1 = collapse1 * (len(negative) // len(collapse1))
                    collapse2 = collapse2 * (len(negative) // len(collapse2))
                elif self.balance == 'tSNE':  # only for tSNE
                    self.scale = [1, 1, len(collapse1) // len(collapse2)]
                    # sample = sorted(random.sample(list(range(len(negative))), 2 * len(collapse1)))
                    with open('dataset/tSNE_idx2.csv', 'r') as f:
                        sample = f.readlines()
                    f.close()
                    negative = [negative[int(item)] for item in sample]
                    collapse2 = collapse2 * (len(collapse1) // len(collapse2))
                    # write_csv(file='dataset/tSNE_idx.csv', tag=[], content=list(map(lambda x: [str(x)], sample)))
                else:
                    raise ValueError
                lines = negative + collapse1 + collapse2

            else:
                raise ValueError

        if self.phase == 'train':
            random.shuffle(lines)
        elif self.phase == 'val' or self.phase == 'test' or self.phase == 'test_train':
            pass
        else:
            raise ValueError

        images = [(str(x[0]).strip().split(',')[0], str(x[1]).strip().split(',')[0], str(x[2]).strip().split(',')[0])
                  for x in tqdm(lines, desc='Preparing Images')]  # 每个元素是三元组
        # 2分类
        if self.num_classes == 2:
            labels = [(0 if int(str(x[0]).strip().split(',')[1]) in [0, 1] else 1,
                       0 if int(str(x[1]).strip().split(',')[1]) in [0, 1] else 1,
                       0 if int(str(x[2]).strip().split(',')[1]) in [0, 1] else 1)
                      for x in tqdm(lines, desc='Preparing Labels')]
        # 3分类
        elif self.num_classes == 3:
            labels = [(0 if int(str(x[0]).strip().split(',')[1]) in [0, 1] else int(str(x[0]).strip().split(',')[1])-1,
                       0 if int(str(x[1]).strip().split(',')[1]) in [0, 1] else int(str(x[1]).strip().split(',')[1])-1,
                       0 if int(str(x[2]).strip().split(',')[1]) in [0, 1] else int(str(x[2]).strip().split(',')[1])-1)
                      for x in tqdm(lines, desc='Preparing Labels')]
        else:
            raise ValueError

        return images, labels

    def dist(self):
        dist = {}

        # label三元组统计
        # for l in tqdm(self.labels, desc="Counting data distribution"):
        #     if str(l) in dist.keys():
        #         dist[str(l)] += 1
        #     else:
        #         dist[str(l)] = 1

        # 三种类别的统计
        for l1, l2, l3 in tqdm(self.labels, desc="Counting data distribution"):
            if str(l2) in dist.keys():
                dist[str(l2)] += 1
            else:
                dist[str(l2)] = 1
        return dist


if __name__ == '__main__':
    train_data = ContextVB_Dataset(csv_path=['dataset/train_VB.csv', 'dataset/val_VB.csv'], num_classes=3, phase='train', useRGB=True, usetrans=True, balance='upsample')
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)

    count = 0
    for imag, lab, img_path in tqdm(train_data):
        print(lab[0], lab[1], lab[2])
        print(img_path[0], img_path[1], img_path[2])
        print('====================================')
        img = Image.fromarray(np.uint8(imag[0].squeeze().numpy() * 255).transpose(1, 2, 0))  # img尺寸(3, 224, 224)，Image转换需要尺寸(224, 224) / (224, 224, 3)
        img.save(f'/DB/rhome/bllai/Data/DATA3/Vertebrae/Sagittal/test/{count}-1_{lab[0]}.png')
        img = Image.fromarray(np.uint8(imag[1].squeeze().numpy() * 255).transpose(1, 2, 0))
        img.save(f'/DB/rhome/bllai/Data/DATA3/Vertebrae/Sagittal/test/{count}-2_{lab[1]}.png')
        img = Image.fromarray(np.uint8(imag[2].squeeze().numpy() * 255).transpose(1, 2, 0))
        img.save(f'/DB/rhome/bllai/Data/DATA3/Vertebrae/Sagittal/test/{count}-3_{lab[2]}.png')

        count += 1
        if count == 20:
            raise KeyboardInterrupt

        # tqdm.write(str(img.size))
        # assert img.size == (224, 224) and lab in [0, 1]
        # tqdm.write(f'{img.mean()}, {img.std()}')
        pass

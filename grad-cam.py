# coding: utf-8
import os
import argparse

import cv2
import numpy as np
import torch

from torch.nn import functional, Sequential
from torch.utils.data import DataLoader
from torch.autograd import Variable, Function
from torchvision import models, utils, transforms
from torchnet import meter
from PIL import Image
from tqdm import tqdm

from dataset import VB_Dataset, Dual_Dataset
from models import Vgg16, ResNet18, ShallowVgg


class FeatureExtractor:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs:
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)
        # self.feature_extractor = FeatureExtractor(self.model.layer4, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output


def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input


def show_cam_on_image(img, mask, save_name):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(save_name, np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if not index:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        # print(self.extractor.get_gradients())
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if not index:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_variables=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False, help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='csv', help='Input image path')
    parser.add_argument('--model-path', type=str, help='Load model path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    # Can work with any model, but it assumes that the model has a feature method, and a classifier method,
    # as in the VGG models in torchvision.

    # model = Vgg16(num_classes=2)
    model = ResNet18(num_classes=3)
    # model = ShallowVgg(num_classes=2)
    model.load(args.model_path)
    model.addfeatures()
    model.eval()

    grad_cam = GradCam(model=model, target_layer_names=["7"], use_cuda=args.use_cuda)  # for resnet
    # grad_cam = GradCam(model=model, target_layer_names=["23"], use_cuda=args.use_cuda)
    # grad_cam = GradCam(model=model, target_layer_names=["16"], use_cuda=args.use_cuda)

    if args.image_path == 'csv':  # 可以直接输入包含多张图片的csv文件
        root = '/DB/rhome/bllai/PyTorchProjects/Vertebrae_Collapse'
        test_paths = [os.path.join(root, 'dataset/test_VB.csv')]
        test_data = VB_Dataset(test_paths, num_classes=3, phase='test', useRGB=True, usetrans=True, padding=True, balance='upsample')
        test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

        test_cm = meter.ConfusionMeter(3)
        test_mAP = meter.mAPMeter()
        softmax = functional.softmax

        for image, label, image_path in tqdm(test_dataloader):
            image.requires_grad = True

            if args.use_cuda:
                model.cuda()
                image = image.cuda()

            score = model(image)

            prob = softmax(score, dim=1).detach().cpu().numpy()[0]
            one_hot = torch.zeros(label.size(0), 3).scatter_(1, label.data.cpu().unsqueeze(1), 1)
            test_cm.add(softmax(score, dim=1).data, label.data)
            test_mAP.add(softmax(score, dim=1).data, one_hot)

            # If None, returns the map for the highest scoring category. Otherwise, targets the requested index.
            # target_index = None
            target_index = 1

            image_path = image_path[0]
            img = cv2.imread(image_path, 1)  # 这一步重新读入图片是为了将CAM图画在原图上
            # img = np.float32(cv2.resize(img, (224, 224))) / 255

            img_split_path = image_path.split('/')
            cam_save_path = '/'.join([*img_split_path[:8], 'CAM_Collapse', model.model_name,
                                      f'CAM_{target_index}_' + args.model_path.split('/')[-1][:-4] + '-' + img_split_path[8], *img_split_path[9:11]])

            if not os.path.exists(cam_save_path):
                os.makedirs(cam_save_path)
            cam_save_name = img_split_path[-1][:-5] + str(label.numpy()[0]) + '_' + str(int(np.argmax(prob))) + '_' + str(round(prob[1], 4)) + '.png'

            if not os.path.exists(os.path.join(cam_save_path, cam_save_name)):
                mask = grad_cam(image, target_index)  # 考虑到upsample有重复的图片，为了节省时间，重复的图片不计算CAM
                show_cam_on_image(img, mask, os.path.join(cam_save_path, cam_save_name))

        print('mAP:', test_mAP.value().numpy())

    else:
        # img = cv2.imread(args.image_path, 1)
        # img = np.float32(cv2.resize(img, (224, 224))) / 255
        img = Image.open(args.image_path)
        save_name = '_'.join(args.image_path.split('/')[-2:])
        if not os.path.exists(os.path.join('results', 'cam')):
            os.makedirs(os.path.join('results', 'cam'))
        img.save(os.path.join('results', 'cam', save_name))
        input = preprocess_image(img)

        model.eval()
        if args.use_cuda:
            model.cuda()
            input = input.cuda()
        softmax = functional.softmax
        print(softmax(model(input), dim=1).data)

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        target_index = None

        mask = grad_cam(input, target_index)

        img = cv2.imread(args.image_path, 1)
        img = np.float32(cv2.resize(img, (112, 112))) / 255
        show_cam_on_image(img, mask, save_name)

        # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        # gb = gb_model(input, index=target_index)
        # utils.save_image(torch.from_numpy(gb), 'gb.jpg')
        #
        # cam_mask = np.zeros(gb.shape)
        # for i in range(0, gb.shape[0]):
        #     cam_mask[i, :, :] = mask
        #
        # cam_gb = np.multiply(cam_mask, gb)
        # utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')

# coding: utf-8

import os
import fire
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from pprint import pprint
from torch.utils.data import DataLoader
from torch.nn import functional
from torchnet import meter
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

from config import config
from dataset import CollapseDataset, VB_Dataset, Dual_Dataset, ContextVB_Dataset
from models import ResNet18, ResNet34, ResNet50, SkipResNet18, DensResNet18, GuideResNet18, Vgg16, AlexNet
from models import densenet_collapse, ShallowVgg, DualNet, CustomedNet, ContextResNet18
from models import FocalLoss, LabelSmoothing
from utils import Visualizer, write_csv, write_json, draw_ROC


def train(**kwargs):
    config.parse(kwargs)
    vis = Visualizer(port=2333, env=config.env)
    vis.log('Use config:')
    for k, v in config.__class__.__dict__.items():
        if not k.startswith('__'):
            vis.log(f"{k}: {getattr(config, k)}")

    # prepare data
    train_data = VB_Dataset(config.train_paths, phase='train', useRGB=config.useRGB, usetrans=config.usetrans, padding=config.padding, balance=config.data_balance)
    val_data = VB_Dataset(config.test_paths, phase='val', useRGB=config.useRGB, usetrans=config.usetrans, padding=config.padding, balance=False)
    print('Training Images:', train_data.__len__(), 'Validation Images:', val_data.__len__())
    dist = train_data.dist()
    print('Train Data Distribution:', dist, 'Val Data Distribution:', val_data.dist())

    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # prepare model
    # model = ResNet18(num_classes=config.num_classes)
    # model = Vgg16(num_classes=config.num_classes)
    # model = densenet_collapse(num_classes=config.num_classes)
    model = ShallowVgg(num_classes=config.num_classes)
    print(model)

    if config.load_model_path:
        model.load(config.load_model_path)
    if config.use_gpu:
        model.cuda()
    if config.parallel:
        model = torch.nn.DataParallel(model, device_ids=[x for x in range(config.num_of_gpu)])

    # criterion and optimizer
    # weight = torch.Tensor([1/dist['0'], 1/dist['1'], 1/dist['2'], 1/dist['3']])
    # weight = torch.Tensor([1/dist['0'], 1/dist['1']])
    # weight = torch.Tensor([dist['1'], dist['0']])
    # weight = torch.Tensor([1, 10])
    # vis.log(f'loss weight: {weight}')
    # print('loss weight:', weight)
    # weight = weight.cuda()

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = LabelSmoothing(size=config.num_classes, smoothing=0.1)
    # criterion = torch.nn.CrossEntropyLoss(weight=weight)
    # criterion = FocalLoss(gamma=4, alpha=None)

    lr = config.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    # metric
    softmax = functional.softmax
    log_softmax = functional.log_softmax
    loss_meter = meter.AverageValueMeter()
    epoch_loss = meter.AverageValueMeter()
    train_cm = meter.ConfusionMeter(config.num_classes)
    train_AUC = meter.AUCMeter()

    previous_avgse = 0
    # previous_AUC = 0
    if config.parallel:
        save_model_dir = config.save_model_dir if config.save_model_dir else model.module.model_name
        save_model_name = config.save_model_name if config.save_model_name else model.module.model_name + '_best_model.pth'
    else:
        save_model_dir = config.save_model_dir if config.save_model_dir else model.model_name
        save_model_name = config.save_model_name if config.save_model_name else model.model_name + '_best_model.pth'
    save_epoch = 1  # 用于记录验证集上效果最好模型对应的epoch
    # process_record = {'epoch_loss': [],  # 用于记录实验过程中的曲线，便于画曲线图
    #                   'train_avgse': [], 'train_se0': [], 'train_se1': [], 'train_se2': [], 'train_se3': [],
    #                   'val_avgse': [], 'val_se0': [], 'val_se1': [], 'val_se2': [], 'val_se3': []}
    process_record = {'epoch_loss': [],  # 用于记录实验过程中的曲线，便于画曲线图
                      'train_avgse': [], 'train_se0': [], 'train_se1': [],
                      'val_avgse': [], 'val_se0': [], 'val_se1': [],
                      'train_AUC': [], 'val_AUC': []}

    # train
    for epoch in range(config.max_epoch):
        print(f"epoch: [{epoch+1}/{config.max_epoch}] {config.save_model_name[:-4]} ==================================")
        epoch_loss.reset()
        train_cm.reset()
        train_AUC.reset()

        # train
        model.train()
        for i, (image, label, image_path) in tqdm(enumerate(train_dataloader)):
            loss_meter.reset()

            # prepare input
            if config.use_gpu:
                image = image.cuda()
                label = label.cuda()

            # go through the model
            score = model(image)

            # backpropagate
            optimizer.zero_grad()
            # loss = criterion(score, label)
            loss = criterion(log_softmax(score, dim=1), label)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())
            epoch_loss.add(loss.item())
            train_cm.add(softmax(score, dim=1).data, label.data)
            positive_score = np.array([item[1] for item in softmax(score, dim=1).data.cpu().numpy().tolist()])
            train_AUC.add(positive_score, label.data)

            if (i+1) % config.print_freq == 0:
                vis.plot('loss', loss_meter.value()[0])

        # print result
        # train_se = [100. * train_cm.value()[0][0] / (train_cm.value()[0][0] + train_cm.value()[0][1] + train_cm.value()[0][2] + train_cm.value()[0][3]),
        #             100. * train_cm.value()[1][1] / (train_cm.value()[1][0] + train_cm.value()[1][1] + train_cm.value()[1][2] + train_cm.value()[1][3]),
        #             100. * train_cm.value()[2][2] / (train_cm.value()[2][0] + train_cm.value()[2][1] + train_cm.value()[2][2] + train_cm.value()[2][3]),
        #             100. * train_cm.value()[3][3] / (train_cm.value()[3][0] + train_cm.value()[3][1] + train_cm.value()[3][2] + train_cm.value()[3][3])]
        train_se = [100. * train_cm.value()[0][0] / (train_cm.value()[0][0] + train_cm.value()[0][1]),
                    100. * train_cm.value()[1][1] / (train_cm.value()[1][0] + train_cm.value()[1][1])]

        # validate
        model.eval()
        if (epoch + 1) % 1 == 0:
            val_cm, val_se, val_accuracy, val_AUC = val_2class(model, val_dataloader)

            if np.average(val_se) > previous_avgse:  # 当测试集上的平均sensitivity升高时保存模型
            # if val_AUC.value()[0] > previous_AUC:  # 当测试集上的AUC升高时保存模型
                if config.parallel:
                    if not os.path.exists(os.path.join('checkpoints', save_model_dir, save_model_name.split('.')[0])):
                        os.makedirs(os.path.join('checkpoints', save_model_dir, save_model_name.split('.')[0]))
                    model.module.save(os.path.join('checkpoints', save_model_dir, save_model_name.split('.')[0], save_model_name))
                else:
                    if not os.path.exists(os.path.join('checkpoints', save_model_dir, save_model_name.split('.')[0])):
                        os.makedirs(os.path.join('checkpoints', save_model_dir, save_model_name.split('.')[0]))
                    model.save(os.path.join('checkpoints', save_model_dir, save_model_name.split('.')[0], save_model_name))
                previous_avgse = np.average(val_se)
                # previous_AUC = val_AUC.value()[0]
                save_epoch = epoch + 1

            process_record['epoch_loss'].append(epoch_loss.value()[0])
            process_record['train_avgse'].append(np.average(train_se))
            process_record['train_se0'].append(train_se[0])
            process_record['train_se1'].append(train_se[1])
            # process_record['train_se2'].append(train_se[2])
            # process_record['train_se3'].append(train_se[3])
            process_record['train_AUC'].append(train_AUC.value()[0])
            process_record['val_avgse'].append(np.average(val_se))
            process_record['val_se0'].append(val_se[0])
            process_record['val_se1'].append(val_se[1])
            # process_record['val_se2'].append(val_se[2])
            # process_record['val_se3'].append(val_se[3])
            process_record['val_AUC'].append(val_AUC.value()[0])

            # vis.plot_many({'epoch_loss': epoch_loss.value()[0],
            #                'train_avgse': np.average(train_se), 'train_se0': train_se[0], 'train_se1': train_se[1], 'train_se2': train_se[2], 'train_se3': train_se[3],
            #                'val_avgse': np.average(val_se), 'val_se0': val_se[0], 'val_se1': val_se[1], 'val_se2': val_se[2], 'val_se3': val_se[3]})
            # vis.log(f"epoch: [{epoch+1}/{config.max_epoch}] =========================================")
            # vis.log(f"lr: {optimizer.param_groups[0]['lr']}, loss: {round(loss_meter.value()[0], 5)}")
            # vis.log(f"train_avgse: {round(np.average(train_se), 4)}, train_se0: {round(train_se[0], 4)}, train_se1: {round(train_se[1], 4)}, train_se2: {round(train_se[2], 4)}, train_se3: {round(train_se[3], 4)},")
            # vis.log(f"val_avgse: {round(np.average(val_se), 4)}, val_se0: {round(val_se[0], 4)}, val_se1: {round(val_se[1], 4)}, val_se2: {round(val_se[2], 4)}, val_se3: {round(val_se[3], 4)}")
            # vis.log(f'train_cm: {train_cm.value()}')
            # vis.log(f'val_cm: {val_cm.value()}')
            # print("lr:", optimizer.param_groups[0]['lr'], "loss:", round(epoch_loss.value()[0], 5))
            # print('train_avgse:', round(np.average(train_se), 4), 'train_se0:', round(train_se[0], 4), 'train_se1:', round(train_se[1], 4), 'train_se2:', round(train_se[2], 4), 'train_se3:', round(train_se[3], 4))
            # print('val_avgse:', round(np.average(val_se), 4), 'val_se0:', round(val_se[0], 4), 'val_se1:', round(val_se[1], 4), 'val_se2:', round(val_se[2], 4), 'val_se3:', round(val_se[3], 4))
            # print('train_cm:')
            # print(train_cm.value())
            # print('val_cm:')
            # print(val_cm.value())

            vis.plot_many({'epoch_loss': epoch_loss.value()[0],
                           'train_avgse': np.average(train_se), 'train_se0': train_se[0], 'train_se1': train_se[1],
                           'val_avgse': np.average(val_se), 'val_se0': val_se[0], 'val_se1': val_se[1],
                           'train_AUC': train_AUC.value()[0], 'val_AUC': val_AUC.value()[0]})
            vis.log(f"epoch: [{epoch + 1}/{config.max_epoch}] =========================================")
            vis.log(f"lr: {optimizer.param_groups[0]['lr']}, loss: {round(loss_meter.value()[0], 5)}")
            vis.log(f"train_avgse: {round(np.average(train_se), 4)}, train_se0: {round(train_se[0], 4)}, train_se1: {round(train_se[1], 4)}")
            vis.log(f"val_avgse: {round(np.average(val_se), 4)}, val_se0: {round(val_se[0], 4)}, val_se1: {round(val_se[1], 4)}")
            vis.log(f'train_AUC: {train_AUC.value()[0]}')
            vis.log(f'val_AUC: {val_AUC.value()[0]}')
            vis.log(f'train_cm: {train_cm.value()}')
            vis.log(f'val_cm: {val_cm.value()}')
            print("lr:", optimizer.param_groups[0]['lr'], "loss:", round(epoch_loss.value()[0], 5))
            print('train_avgse:', round(np.average(train_se), 4), 'train_se0:', round(train_se[0], 4), 'train_se1:', round(train_se[1], 4))
            print('val_avgse:', round(np.average(val_se), 4), 'val_se0:', round(val_se[0], 4), 'val_se1:', round(val_se[1], 4))
            print('train_AUC:', train_AUC.value()[0], 'val_AUC:', val_AUC.value()[0])
            print('train_cm:')
            print(train_cm.value())
            print('val_cm:')
            print(val_cm.value())

            if os.path.exists(os.path.join('checkpoints', save_model_dir, save_model_name.split('.')[0])):
                write_json(file=os.path.join('checkpoints', save_model_dir, save_model_name.split('.')[0], 'process_record.json'), content=process_record)

        # if (epoch+1) % 5 == 0:
        #     lr = lr * config.lr_decay
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr

    vis.log(f"Best Epoch: {save_epoch}")
    print("Best Epoch:", save_epoch)


def iter_train(**kwargs):
    config.parse(kwargs)

    # ============================================ Visualization =============================================
    # vis = Visualizer(port=2333, env=config.env)
    # vis.log('Use config:')
    # for k, v in config.__class__.__dict__.items():
    #     if not k.startswith('__'):
    #         vis.log(f"{k}: {getattr(config, k)}")

    # ============================================= Prepare Data =============================================
    train_data = VB_Dataset(config.train_paths, phase='train', num_classes=config.num_classes, useRGB=config.useRGB,
                            usetrans=config.usetrans, padding=config.padding, balance=config.data_balance)
    val_data = VB_Dataset(config.test_paths, phase='val', num_classes=config.num_classes, useRGB=config.useRGB,
                          usetrans=config.usetrans, padding=config.padding, balance=config.data_balance)
    train_dist, val_dist = train_data.dist(), val_data.dist()
    train_data_scale, val_data_scale = train_data.scale, val_data.scale
    print('Training Images:', train_data.__len__(), 'Validation Images:', val_data.__len__())
    print('Train Data Distribution:', train_dist, 'Val Data Distribution:', val_dist)

    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # ============================================= Prepare Model ============================================
    # model = ResNet18(num_classes=config.num_classes)
    # model = ResNet34(num_classes=config.num_classes)
    # model = ResNet50(num_classes=config.num_classes)
    model = Vgg16(num_classes=config.num_classes)
    # model = AlexNet(num_classes=config.num_classes)
    # model = densenet_collapse(num_classes=config.num_classes)
    # model = ShallowVgg(num_classes=config.num_classes)
    # model = CustomedNet(num_classes=config.num_classes)
    # model = DualNet(num_classes=config.num_classes)
    # model = SkipResNet18(num_classes=config.num_classes)
    # model = DensResNet18(num_classes=config.num_classes)
    # model = GuideResNet18(num_classes=config.num_classes)
    # print(model)

    if config.load_model_path:
        model.load(config.load_model_path)
    if config.use_gpu:
        model.cuda()
    if config.parallel:
        model = torch.nn.DataParallel(model, device_ids=list(range(config.num_of_gpu)))

    # =========================================== Criterion and Optimizer =====================================
    # weight = torch.Tensor([1/dist['0'], 1/dist['1'], 1/dist['2'], 1/dist['3']])
    # weight = torch.Tensor([1/dist['0'], 1/dist['1']])
    # weight = torch.Tensor([dist['1'], dist['0']])
    # weight = torch.Tensor([1, 10])
    # vis.log(f'loss weight: {weight}')
    # print('loss weight:', weight)
    # weight = weight.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.CrossEntropyLoss(weight=weight)
    # criterion = LabelSmoothing(size=config.num_classes, smoothing=0.2)
    # criterion = FocalLoss(gamma=4, alpha=None)

    lr = config.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    # ================================================== Metrics ===============================================
    log_softmax = functional.log_softmax
    loss_meter = meter.AverageValueMeter()

    # ====================================== Saving and Recording Configuration =================================
    previous_AUC = 0
    previous_mAP = 0
    save_iter = 1  # 用于记录验证集上效果最好模型对应的epoch
    if config.parallel:
        save_model_dir = config.save_model_dir if config.save_model_dir else model.module.model_name
        save_model_name = config.save_model_name if config.save_model_name else model.module.model_name + '_best_model.pth'
    else:
        save_model_dir = config.save_model_dir if config.save_model_dir else model.model_name
        save_model_name = config.save_model_name if config.save_model_name else model.model_name + '_best_model.pth'
    if config.num_classes == 2:  # 2分类
        process_record = {'loss': [],  # 用于记录实验过程中的曲线，便于画曲线图
                          'train_avg': [], 'train_sp': [], 'train_se': [],
                          'val_avg': [], 'val_sp': [], 'val_se': [],
                          'train_AUC': [], 'val_AUC': []}
    elif config.num_classes == 3:  # 3分类
        process_record = {'loss': [],  # 用于记录实验过程中的曲线，便于画曲线图
                          'train_sp0': [], 'train_se0': [], 'train_sp1': [], 'train_se1': [], 'train_sp2': [], 'train_se2': [],
                          'val_sp0': [], 'val_se0': [], 'val_sp1': [], 'val_se1': [], 'val_sp2': [], 'val_se2': [],
                          'train_mAUC': [], 'val_mAUC': [], 'train_mAP': [], 'val_mAP': []}
    else:
        raise ValueError

    # ================================================== Training ===============================================
    iteration = 0
    # ****************************************** train ****************************************
    train_iter = iter(train_dataloader)
    model.train()
    while iteration < config.max_iter:
        try:
            image, label, image_path = next(train_iter)
        except:
            train_iter = iter(train_dataloader)
            image, label, image_path = next(train_iter)

        iteration += 1

        # ------------------------------------ prepare input ------------------------------------
        if config.use_gpu:
            image = image.cuda()
            label = label.cuda()

        # ---------------------------------- go through the model --------------------------------
        score = model(image)

        # ----------------------------------- backpropagate -------------------------------------
        optimizer.zero_grad()
        loss = criterion(score, label)
        # loss = criterion(log_softmax(score, dim=1), label)  # LabelSmoothing
        loss.backward()
        optimizer.step()

        # ------------------------------------ record loss ------------------------------------
        loss_meter.add(loss.item())

        if iteration % config.print_freq == 0:
            tqdm.write(f"iter: [{iteration}/{config.max_iter}] {config.save_model_name[:-4]} ==================================")

            # *************************************** validate ***************************************
            if config.num_classes == 2:  # 2分类
                model.eval()
                train_cm, train_AUC, train_sp, train_se, train_T, train_accuracy = val_2class(model, train_dataloader, train_dist)
                val_cm, val_AUC, val_sp, val_se, val_T, val_accuracy = val_2class(model, val_dataloader, val_dist)
                # vis.plot('loss', loss_meter.value()[0])

                model.train()

                # ------------------------------------ save model ------------------------------------
                if val_AUC > previous_AUC:  # 当测试集上的AUC升高时保存模型
                    if config.parallel:
                        if not os.path.exists(os.path.join('checkpoints', save_model_dir, save_model_name[:-4])):
                            os.makedirs(os.path.join('checkpoints', save_model_dir, save_model_name[:-4]))
                        model.module.save(os.path.join('checkpoints', save_model_dir, save_model_name[:-4], save_model_name))
                    else:
                        if not os.path.exists(os.path.join('checkpoints', save_model_dir, save_model_name[:-4])):
                            os.makedirs(os.path.join('checkpoints', save_model_dir, save_model_name[:-4]))
                        model.save(os.path.join('checkpoints', save_model_dir, save_model_name[:-4], save_model_name))
                    previous_AUC = val_AUC
                    save_iter = iteration

                # ---------------------------------- recond and print ---------------------------------
                process_record['loss'].append(loss_meter.value()[0])
                process_record['train_avg'].append((train_sp + train_se) / 2)
                process_record['train_sp'].append(train_sp)
                process_record['train_se'].append(train_se)
                process_record['train_AUC'].append(train_AUC)
                process_record['val_avg'].append((val_sp + val_se) / 2)
                process_record['val_sp'].append(val_sp)
                process_record['val_se'].append(val_se)
                process_record['val_AUC'].append(val_AUC)

                # vis.plot_many({'loss': loss_meter.value()[0],
                #                'train_avg': (train_sp + train_se) / 2, 'train_sp': train_sp, 'train_se': train_se,
                #                'val_avg': (val_sp + val_se) / 2, 'val_sp': val_sp, 'val_se': val_se,
                #                'train_AUC': train_AUC, 'val_AUC': val_AUC})
                # vis.log(f"iter: [{iteration}/{config.max_iter}] =========================================")
                # vis.log(f"lr: {optimizer.param_groups[0]['lr']}, loss: {round(loss_meter.value()[0], 5)}")
                # vis.log(f"train_avg: {round((train_sp + train_se) / 2, 4)}, train_sp: {round(train_sp, 4)}, train_se: {round(train_se, 4)}")
                # vis.log(f"val_avg: {round((val_sp + val_se) / 2, 4)}, val_sp: {round(val_sp, 4)}, val_se: {round(val_se, 4)}")
                # vis.log(f'train_AUC: {train_AUC}')
                # vis.log(f'val_AUC: {val_AUC}')
                # vis.log(f'train_cm: {train_cm}')
                # vis.log(f'val_cm: {val_cm}')
                print("lr:", optimizer.param_groups[0]['lr'], "loss:", round(loss_meter.value()[0], 5))
                print('train_avg:', round((train_sp + train_se) / 2, 4), 'train_sp:', round(train_sp, 4), 'train_se:', round(train_se, 4))
                print('val_avg:', round((val_sp + val_se) / 2, 4), 'val_sp:', round(val_sp, 4), 'val_se:', round(val_se, 4))
                print('train_AUC:', train_AUC, 'val_AUC:', val_AUC)
                print('train_cm:')
                print(train_cm)
                print('val_cm:')
                print(val_cm)

            elif config.num_classes == 3:  # 3分类
                model.eval()
                train_cm, train_mAP, train_sp, train_se, train_mAUC, train_accuracy = val_3class(model, train_dataloader, train_data_scale)
                val_cm, val_mAP, val_sp, val_se, val_mAUC, val_accuracy = val_3class(model, val_dataloader, val_data_scale)
                model.train()

                # ------------------------------------ save model ------------------------------------
                if val_mAP > previous_mAP:  # 当测试集上的mAP升高时保存模型
                    if config.parallel:
                        if not os.path.exists(os.path.join('checkpoints', save_model_dir, save_model_name[:-4])):
                            os.makedirs(os.path.join('checkpoints', save_model_dir, save_model_name[:-4]))
                        model.module.save(os.path.join('checkpoints', save_model_dir, save_model_name[:-4], save_model_name))
                    else:
                        if not os.path.exists(os.path.join('checkpoints', save_model_dir, save_model_name[:-4])):
                            os.makedirs(os.path.join('checkpoints', save_model_dir, save_model_name[:-4]))
                        model.save(os.path.join('checkpoints', save_model_dir, save_model_name[:-4], save_model_name))
                    previous_mAP = val_mAP
                    save_iter = iteration

                # ---------------------------------- recond and print ---------------------------------
                process_record['loss'].append(loss_meter.value()[0])
                process_record['train_sp0'].append(train_sp[0])
                process_record['train_se0'].append(train_se[0])
                process_record['train_sp1'].append(train_sp[1])
                process_record['train_se1'].append(train_se[1])
                process_record['train_sp2'].append(train_sp[2])
                process_record['train_se2'].append(train_se[2])
                process_record['train_mAUC'].append(float(train_mAUC))
                process_record['train_mAP'].append(float(train_mAP))
                process_record['val_sp0'].append(val_sp[0])
                process_record['val_se0'].append(val_se[0])
                process_record['val_sp1'].append(val_sp[1])
                process_record['val_se1'].append(val_se[1])
                process_record['val_sp2'].append(val_sp[2])
                process_record['val_se2'].append(val_se[2])
                process_record['val_mAUC'].append(float(val_mAUC))
                process_record['val_mAP'].append(float(val_mAP))

                # vis.plot_many({'loss': loss_meter.value()[0],
                #                'train_sp0': train_se[0], 'train_sp1': train_se[1], 'train_sp2': train_se[2],
                #                'train_se0': train_se[0], 'train_se1': train_se[1], 'train_se2': train_se[2],
                #                'val_sp0': val_se[0], 'val_sp1': val_se[1], 'val_sp2': val_se[2],
                #                'val_se0': val_se[0], 'val_se1': val_se[1], 'val_se2': val_se[2],
                #                'train_mAP': train_mAP, 'val_mAP': val_mAP})
                # vis.log(f"iter: [{iteration}/{config.max_iter}] =========================================")
                # vis.log(f"lr: {optimizer.param_groups[0]['lr']}, loss: {round(loss_meter.value()[0], 5)}")
                # vis.log(f"train_sp0: {round(train_sp[0], 4)}, train_sp1: {round(train_sp[1], 4)}, train_sp2: {round(train_sp[2], 4)}")
                # vis.log(f"train_se0: {round(train_se[0], 4)}, train_se1: {round(train_se[1], 4)}, train_se2: {round(train_se[2], 4)}")
                # vis.log(f"val_sp0: {round(val_sp[0], 4)}, val_sp1: {round(val_sp[1], 4)}, val_sp2: {round(val_sp[2], 4)}")
                # vis.log(f"val_se0: {round(val_se[0], 4)}, val_se1: {round(val_se[1], 4)}, val_se2: {round(val_se[2], 4)}")
                # vis.log(f"train_mAP: {train_mAP}, val_mAP: {val_mAP}")
                # vis.log(f'train_cm: {train_cm}')
                # vis.log(f'val_cm: {val_cm}')
                print("lr:", optimizer.param_groups[0]['lr'], "loss:", round(loss_meter.value()[0], 5))
                print('train_sp0:', round(train_sp[0], 4), 'train_sp1:', round(train_sp[1], 4), 'train_sp2:', round(train_sp[2], 4))
                print('train_se0:', round(train_se[0], 4), 'train_se1:', round(train_se[1], 4), 'train_se2:', round(train_se[2], 4))
                print('val_sp0:', round(val_sp[0], 4), 'val_sp1:', round(val_sp[1], 4), 'val_sp2:', round(val_sp[2], 4))
                print('val_se0:', round(val_se[0], 4), 'val_se1:', round(val_se[1], 4), 'val_se2:', round(val_se[2], 4))
                print('mSP:', round(sum(val_sp)/3, 5), 'mSE:', round(sum(val_se)/3, 5))
                print('train_mAUC:', train_mAUC, 'val_mAUC:', val_mAUC)
                print('train_mAP:', train_mAP, 'val_mAP:', val_mAP)
                print('train_cm:')
                print(train_cm)
                print('val_cm:')
                print(val_cm)
                print('Best mAP:', previous_mAP)

            loss_meter.reset()

        # ------------------------------------ save record ------------------------------------
        if os.path.exists(os.path.join('checkpoints', save_model_dir, save_model_name.split('.')[0])):
            write_json(file=os.path.join('checkpoints', save_model_dir, save_model_name.split('.')[0], 'process_record.json'), content=process_record)

    # vis.log(f"Best Iter: {save_iter}")
    print("Best Iter:", save_iter)


def val_2class(model, dataloader, dist):
    # ============================ Prepare Metrics ==========================
    y_true, y_scores = [], []

    softmax = functional.softmax

    # ================================ Validate ==============================
    for i, (image, label, image_path) in tqdm(enumerate(dataloader)):

        # ******************* prepare input and go through the model *******************
        if config.use_gpu:
            image = image.cuda()
            label = label.cuda()
        image.requires_grad = False
        label.requires_grad = False

        score = model(image)

        # *********************** confusion matrix and AUC ***********************
        positive_score = [item[1] for item in softmax(score, dim=1).data.cpu().numpy().tolist()]
        y_true.extend(label.data.cpu().numpy().tolist())  # 用于sklearn计算AUC和ROC
        y_scores.extend(positive_score)

    # ************************** TPR, FPR, AUC ******************************
    SKL_FPR, SKL_TPR, SKL_Thresholds = roc_curve(y_true, y_scores)
    SKL_AUC = roc_auc_score(np.array(y_true), np.array(y_scores), average='weighted')

    # ******************** Best SE, SP, Thresh, Matrix ***********************
    best_index = np.argmax(SKL_TPR - SKL_FPR, axis=0)
    best_SE, best_SP, best_T = SKL_TPR[best_index], 1 - SKL_FPR[best_index], SKL_Thresholds[best_index]
    best_confusion_matrix = [[int(round(dist['0'] * best_SP)), int(round(dist['0'] * (1 - best_SP)))],
                             [int(round(dist['1'] * (1 - best_SE))), int(round(dist['1'] * best_SE))]]

    # *********************** accuracy and sensitivity ***********************
    val_accuracy = 100. * sum([best_confusion_matrix[c][c] for c in range(config.num_classes)]) / np.sum(best_confusion_matrix)
    # val_se = [100. * best_confusion_matrix[i][i] / np.sum(best_confusion_matrix[i]) for i in range(config.num_classes)]

    return best_confusion_matrix, SKL_AUC, best_SP, best_SE, best_T, val_accuracy


def val_3class(model, dataloader, data_scale):
    # ============================ Prepare Metrics ==========================
    val_cm = meter.ConfusionMeter(config.num_classes)
    val_mAP = meter.mAPMeter()
    y_true_0, y_scores_0 = [], []
    y_true_1, y_scores_1 = [], []
    y_true_2, y_scores_2 = [], []

    softmax = functional.softmax

    # ================================ Validate ==============================
    for i, (image, label, image_path) in tqdm(enumerate(dataloader)):

        # ******************* prepare input and go through the model *******************
        if config.use_gpu:
            image = image.cuda()
            label = label.cuda()

        image.requires_grad = False
        label.requires_grad = False

        score = model(image)

        # *********************** confusion matrix and mAP ***********************
        one_hot = torch.zeros(label.size(0), 3).scatter_(1, label.data.cpu().unsqueeze(1), 1)

        val_cm.add(softmax(score, dim=1).data, label.data)
        val_mAP.add(softmax(score, dim=1).data, one_hot)

        positive_score_0 = [item[0] for item in softmax(score, dim=1).data.cpu().numpy().tolist()]
        positive_score_1 = [item[1] for item in softmax(score, dim=1).data.cpu().numpy().tolist()]
        positive_score_2 = [item[2] for item in softmax(score, dim=1).data.cpu().numpy().tolist()]
        label_0 = [1 if item == 0 else 0 for item in label.data.cpu().numpy().tolist()]
        label_1 = [1 if item == 1 else 0 for item in label.data.cpu().numpy().tolist()]
        label_2 = [1 if item == 2 else 0 for item in label.data.cpu().numpy().tolist()]

        y_true_0.extend(label_0)  # 用于sklearn计算AUC和ROC
        y_true_1.extend(label_1)
        y_true_2.extend(label_2)
        y_scores_0.extend(positive_score_0)
        y_scores_1.extend(positive_score_1)
        y_scores_2.extend(positive_score_2)

    # *********************** accuracy and sensitivity ***********************
    AUC_0 = roc_auc_score(np.array(y_true_0), np.array(y_scores_0), average='weighted')
    AUC_1 = roc_auc_score(np.array(y_true_1), np.array(y_scores_1), average='weighted')
    AUC_2 = roc_auc_score(np.array(y_true_2), np.array(y_scores_2), average='weighted')

    val_cm = val_cm.value()
    val_accuracy = 100. * sum([val_cm[c][c] for c in range(config.num_classes)]) / val_cm.sum()
    val_sp = [100. * (val_cm.sum() - val_cm.sum(0)[i] - val_cm.sum(1)[i] + val_cm[i][i]) / (val_cm.sum() - val_cm.sum(1)[i])
              for i in range(config.num_classes)]
    val_se = [100. * val_cm[i][i] / val_cm.sum(1)[i] for i in range(config.num_classes)]
    val_cm = val_cm / np.expand_dims(np.array(data_scale), axis=1)  # 计算指标时按照balance后的matrix来算，展示的时候还原

    return val_cm.astype(dtype=np.int32), val_mAP.value().numpy(), val_sp, val_se, (AUC_0+AUC_1+AUC_2)/3, val_accuracy


def test_2class(**kwargs):
    config.parse(kwargs)

    # ============================================= Prepare Data =============================================
    test_data = VB_Dataset(config.test_paths, phase='test', num_classes=config.num_classes, useRGB=config.useRGB,
                           usetrans=config.usetrans, padding=config.padding, balance=False)
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_dist = test_data.dist()

    print('Test Image:', test_data.__len__())

    # ============================================= Prepare Model ============================================
    # model = ResNet18(num_classes=config.num_classes)
    model = ShallowVgg(num_classes=config.num_classes)
    print(model)

    if config.load_model_path:
        model.load(config.load_model_path)
        print('Model has been loaded!')
    else:
        print("Don't load model")
    if config.use_gpu:
        model.cuda()
    if config.parallel:
        model = torch.nn.DataParallel(model, device_ids=[x for x in range(config.num_of_gpu)])
    model.eval()

    # =========================================== Prepare Metrics =====================================
    test_cm = meter.ConfusionMeter(config.num_classes)
    test_AUC = meter.AUCMeter()
    softmax = functional.softmax
    results = []
    y_true, y_scores = [], []

    # =========================================== Test ============================================
    for i, (image, label, image_path) in tqdm(enumerate(test_dataloader)):
        # ******************* prepare input and go through the model *******************
        if config.use_gpu:
            image = image.cuda()
            label = label.cuda()
        image.requires_grad = False
        label.requires_grad = False

        score = model(image)

        # *************************** confusion matrix and AUC *************************
        test_cm.add(softmax(score, dim=1).data, label.data)
        positive_score = np.array([item[1] for item in softmax(score, dim=1).data.cpu().numpy().tolist()])
        test_AUC.add(positive_score, label.data)  # torchnet计算AUC和ROC

        y_true.extend(label.data.cpu().numpy().tolist())  # 用于sklearn计算AUC和ROC
        y_scores.extend(positive_score.tolist())

        # ******************************** record prediction results ******************************
        for l, p, ip in zip(label.detach(), softmax(score, dim=1).detach(), image_path):
            if p[1] < 0.5:
                results.append((ip, int(l), 0, round(float(p[0]), 4), round(float(p[1]), 4)))
            else:
                results.append((ip, int(l), 1, round(float(p[0]), 4), round(float(p[1]), 4)))

    # ************************** TPR, FPR, AUC ******************************
    SKL_FPR, SKL_TPR, SKL_Thresholds = roc_curve(y_true, y_scores)
    SKL_AUC = roc_auc_score(np.array(y_true), np.array(y_scores), average='weighted')

    TNet_AUC, TNet_TPR, TNet_FPR = test_AUC.value()

    # ******************** Best SE, SP, Thresh, Matrix ***********************
    best_index = np.argmax(SKL_TPR - SKL_FPR, axis=0)
    best_SE, best_SP, best_T = SKL_TPR[best_index], 1-SKL_FPR[best_index], SKL_Thresholds[best_index]
    best_confusion_matrix = [[int(round(test_dist['0'] * best_SP)), int(round(test_dist['0'] * (1 - best_SP)))],
                             [int(round(test_dist['1'] * (1 - best_SE))), int(round(test_dist['1'] * best_SE))]]

    # *********************** accuracy and sensitivity ***********************
    test_accuracy = 100. * sum([test_cm.value()[c][c] for c in range(config.num_classes)]) / np.sum(test_cm.value())
    test_se = [100. * test_cm.value()[i][i] / np.sum(test_cm.value()[i]) for i in range(config.num_classes)]

    # ================================ Save and Print Prediction Results ===========================
    if config.result_file:
        write_csv(os.path.join('results', config.result_file), tag=['path', 'label', 'predict', 'p1', 'p2'], content=results)

    draw_ROC(tpr=SKL_TPR, fpr=SKL_FPR, best_index=best_index, tangent=True,
             save_path=os.path.join('results', config.load_model_path.split('/')[-1][:-4] + "_ROC.png"))

    print('test_acc:', test_accuracy)
    print('test_avgse:', round(np.average(test_se), 4), 'train_se0:', round(test_se[0], 4), 'train_se1:', round(test_se[1], 4))
    print('SKL_AUC:', SKL_AUC, 'TNet_AUC:', TNet_AUC)
    print('Best_SE:', best_SE, 'Best_SP:', best_SP, 'Best_Threshold:', best_T)
    print('test_cm:')
    print(best_confusion_matrix)


def test_3class(**kwargs):
    config.parse(kwargs)

    # ============================================= Prepare Data =============================================
    test_data = VB_Dataset(config.test_paths, phase='test', num_classes=config.num_classes, useRGB=config.useRGB,
                           usetrans=config.usetrans, padding=config.padding, balance=config.data_balance)
    test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    test_dist, test_scale = test_data.dist(), test_data.scale

    print('Test Image:', test_data.__len__())
    print('Test Data Distribution:', test_dist)

    # ============================================= Prepare Model ============================================
    # model = AlexNet(num_classes=config.num_classes)
    # model = Vgg16(num_classes=config.num_classes)
    model = ResNet18(num_classes=config.num_classes)
    # model = ResNet50(num_classes=config.num_classes)
    # print(model)

    if config.load_model_path:
        model.load(config.load_model_path)
        print('Model has been loaded!')
    else:
        print("Don't load model")
    if config.use_gpu:
        model.cuda()
    if config.parallel:
        model = torch.nn.DataParallel(model, device_ids=list(range(config.num_of_gpu)))
    model.eval()

    # ============================ Prepare Metrics ==========================
    test_cm = meter.ConfusionMeter(config.num_classes)
    test_mAP = meter.mAPMeter()
    y_true_0, y_scores_0 = [], []
    y_true_1, y_scores_1 = [], []
    y_true_2, y_scores_2 = [], []
    results = []
    features, colors = [], []  # for t-SNE

    softmax = functional.softmax

    # ================================== Test ===============================
    for i, (image, label, image_path) in tqdm(enumerate(test_dataloader)):

        # ******************* prepare input and go through the model *******************
        if config.use_gpu:
            image = image.cuda()
            label = label.cuda()
        image.requires_grad = False
        label.requires_grad = False

        score = model(image)

        # *********************** t-SNE feature and colors ***********************
        features.append(score.detach().cpu().numpy())
        for c in label.cpu().numpy().tolist():
            if c == 0:
                colors.append('springgreen')
            elif c == 1:
                colors.append('mediumblue')
            elif c == 2:
                colors.append('red')
            else:
                raise ValueError

        # *********************** confusion matrix and mAP ***********************
        one_hot = torch.zeros(label.size(0), 3).scatter_(1, label.data.cpu().unsqueeze(1), 1)

        test_cm.add(softmax(score, dim=1).data, label.data)
        test_mAP.add(softmax(score, dim=1).data, one_hot)

        positive_score_0 = [item[0] for item in softmax(score, dim=1).data.cpu().numpy().tolist()]
        positive_score_1 = [item[1] for item in softmax(score, dim=1).data.cpu().numpy().tolist()]
        positive_score_2 = [item[2] for item in softmax(score, dim=1).data.cpu().numpy().tolist()]
        label_0 = [1 if item == 0 else 0 for item in label.data.cpu().numpy().tolist()]
        label_1 = [1 if item == 1 else 0 for item in label.data.cpu().numpy().tolist()]
        label_2 = [1 if item == 2 else 0 for item in label.data.cpu().numpy().tolist()]

        y_true_0.extend(label_0)  # 用于sklearn计算AUC和ROC
        y_true_1.extend(label_1)
        y_true_2.extend(label_2)
        y_scores_0.extend(positive_score_0)
        y_scores_1.extend(positive_score_1)
        y_scores_2.extend(positive_score_2)

        # ******************************** record prediction results ******************************
        for l, p, ip in zip(label.detach(), softmax(score, dim=1).detach(), image_path):
            if p[0] > p[1] and p[0] > p[2]:
                results.append((ip, int(l), 0, round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)))
            elif p[1] > p[0] and p[1] > p[2]:
                results.append((ip, int(l), 1, round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)))
            else:
                results.append((ip, int(l), 2, round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)))

    # ================================== accuracy and sensitivity ==================================
    AUC_0 = roc_auc_score(np.array(y_true_0), np.array(y_scores_0), average='weighted')
    AUC_1 = roc_auc_score(np.array(y_true_1), np.array(y_scores_1), average='weighted')
    AUC_2 = roc_auc_score(np.array(y_true_2), np.array(y_scores_2), average='weighted')

    test_cm = test_cm.value()
    test_accuracy = 100. * sum([test_cm[c][c] for c in range(config.num_classes)]) / test_cm.sum()
    test_sp = [100. * (test_cm.sum() - test_cm.sum(0)[i] - test_cm.sum(1)[i] + test_cm[i][i]) / (test_cm.sum() - test_cm.sum(1)[i])
               for i in range(config.num_classes)]
    test_se = [100. * test_cm[i][i] / test_cm.sum(1)[i] for i in range(config.num_classes)]
    test_cm = test_cm / np.expand_dims(np.array(test_scale), axis=1)  # 计算指标时按照balance后的matrix来算，展示的时候还原

    # ============================================ t-SNE ===========================================
    features = np.concatenate(features, axis=0)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Y = tsne.fit_transform(features)  # 转换后的输出
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(Y[:, 0], Y[:, 1], c=colors, cmap=plt.cm.Spectral)
    ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.savefig(f'results/{config.load_model_path.split("/")[-1][:-4]}_logits2.png')
    # ipdb.set_trace()

    # ================================ Save and Print Prediction Results ===========================
    if config.result_file:
        write_csv(os.path.join('results', config.result_file), tag=['path', 'label', 'predict', 'p1', 'p2', 'p3'], content=results)

    print('test_acc:', test_accuracy)
    print('test_sp0:', test_sp[0], 'test_sp1:', test_sp[1], 'test_sp2:', test_sp[2])
    print('test_se0:', test_se[0], 'test_se1:', test_se[1], 'test_se2:', test_se[2])
    print('mSP:', round(sum(test_sp) / 3, 5), 'mSE:', round(sum(test_se) / 3, 5))
    print('test_mAUC:', (AUC_0 + AUC_1 + AUC_2) / 3)
    print('test_mAP:', test_mAP.value().numpy())
    print('test_cm:')
    print(test_cm.astype(dtype=np.int32))


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'iter_train': iter_train,
        'test_2class': test_2class,
        'test_3class': test_3class
    })

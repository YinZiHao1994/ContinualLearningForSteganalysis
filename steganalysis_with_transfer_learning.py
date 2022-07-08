# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
# import pandas as pd
from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
# 30 SRM filtes
# from srm_filter_kernel import all_normalized_hpf_list
# Global covariance pooling
# from MPNCOV import *  # MPNCOV
from SRNet import SRNet
from enum import Enum
from dataset import MyDataset
import steganalysis_utils
import dataAnalyze
from common import DatasetEnum, SteganographyEnum
import common_utils

BATCH_SIZE = 32
EPOCHS = 60
LR = 0.01
WEIGHT_DECAY = 5e-4
TRAIN_PRINT_FREQUENCY = 50
EVAL_PRINT_FREQUENCY = 10
STETSIZE = 14
scheduler_gama = 0.40
# DECAY_EPOCH = [30, 60, 90, 140, 200, 250, 300, 350]
# DECAY_EPOCH = [20, 60, 90, 120, 150, 170, 190]
DECAY_EPOCH = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
LOG_PATH = 'data/log'
MODEL_EXPORT_PATH = 'data/model_export'
# DATASET_DIR = r'D:\Work\dataset\steganalysis\BOSSBase'
use_gpu = torch.cuda.is_available()


def train_implement(model, device, train_loader, optimizer, epoch, steganography, diagram_data):
    # losses = steganalysis_utils.AverageMeter()
    model.train()
    train_loss = 0
    train_correct = 0
    total = 0

    loss_history, acc_history, counter = diagram_data.loss_history, diagram_data.acc_history, diagram_data.counter,

    for i, sample in enumerate(train_loader):

        datas, labels = sample['data'], sample['label']
        shape = list(datas.size())
        datas = datas.reshape(shape[0] * shape[1], *shape[2:])
        labels = labels.reshape(-1)
        # shuffle
        idx = torch.randperm(shape[0])
        data = datas[idx]
        label = labels[idx]

        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(data)  # FP
        del data
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)
        # losses.update(loss.item(), data.size(0))
        loss.backward()  # BP
        optimizer.step()
        train_loss += loss.item()
        prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
        data_num = label.size(0)
        total += data_num
        corrects = np.sum(prediction[1].cpu().numpy() == label.cpu().numpy())
        train_correct += corrects
        del label

        if i % 10 == 0:
            diagram_data.iteration_number += 10
            counter.append(diagram_data.iteration_number)
            loss_history.append(loss.item())
            acc_history.append(corrects / data_num)

        if i % TRAIN_PRINT_FREQUENCY == 0:
            print('train epoch: [{0}][{1}/{2}]\t'
                  'Acc {acc:.4f}\t'
                  'Loss {loss:.4f} \t'
                  .format(epoch, i, len(train_loader), acc=100. * train_correct / total,
                          loss=train_loss / (i + 1)))
    diagram_save_path = os.path.join(os.getcwd(), "diagram", steganography.name)
    dataAnalyze.save_loss_plot(diagram_save_path, counter, loss_history,
                               "loss_train" + "_" + str(epoch))
    dataAnalyze.save_accurate_plot(diagram_save_path, counter, acc_history,
                                   "acc_train" + "_" + str(epoch))
    return model


def evaluate(model, device, data_loader):
    print('start evaluate')
    model.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0
    batch_num = 0
    best_acc = 0.0

    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            if i % 10 == 0:
                print("evaluate in {}/{}".format(i, len(data_loader)))
            data, label = sample['data'], sample['label']
            shape = list(data.size())
            data = data.reshape(shape[0] * shape[1], *shape[2:])
            label = label.reshape(-1)
            # shuffle
            idx = torch.randperm(shape[0])
            data = data[idx]
            label = label[idx]

            data, label = data.to(device), label.to(device)
            output = model(data)
            del data
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, label)
            test_loss += loss.item()
            batch_num += 1

            pred = output.max(1, keepdim=True)[1]
            total += label.size(0)
            correct += pred.eq(label.view_as(pred)).sum().item()
            del label

    # accuracy = correct / (len(eval_loader.dataset) * 2)
    accuracy = 100. * correct / total

    # if accuracy > best_acc:
    #     best_acc = accuracy
    # all_state = {
    #     'original_state': model.state_dict(),
    #     'optimizer_state': optimizer.state_dict(),
    #     'epoch': epoch
    # }
    # torch.save(all_state, params_path)

    print('-' * 8)
    test_loss = test_loss / batch_num
    print('Evaluate loss: {:.4f}'.format(test_loss))
    print('Eval accuracy: {:.4f}'.format(accuracy))
    # print('Best accuracy:{:.4f}'.format(best_acc))
    print('-' * 8)
    return accuracy, test_loss


# Initialization
def init_weights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
        nn.init.constant_(module.bias.data, val=0)


def train_model(model, target_dataset, target_steganography, train_loader, valid_loader, epochs, lr=0.01):
    """
    训练一种隐写分析算法模型。
    :param epochs:
    :param lr:
    :param model:
    :param train_loader:
    :param valid_loader:
    :param target_dataset: 所使用的数据集
    :param target_steganography:希望分析的隐写算法
    此参数又为空，默认赋值为 @target_steganography 的值
    """
    device = torch.device("cuda" if use_gpu else "cpu")
    # init_logger(target_dataset, target_steganography)

    if not os.path.exists(MODEL_EXPORT_PATH):
        os.makedirs(MODEL_EXPORT_PATH)
    # model_save_file_name = 'SRNET_model_params_boss_256_HILL04.pt'
    target_model_save_file_name = generate_model_save_file_name(target_dataset, target_steganography)
    target_model_save_path = os.path.join(MODEL_EXPORT_PATH, target_model_save_file_name)
    params_save_file_name = 'SRNET_model_params_' + target_dataset.name + '_' + target_steganography.name + '04.tar'
    params_save_file_path = os.path.join(MODEL_EXPORT_PATH, params_save_file_name)

    model.train()
    params = model.parameters()
    params_wd, params_rest = [], []
    for param_item in params:
        if param_item.requires_grad:
            (params_wd if param_item.dim() != 1 else params_rest).append(param_item)
    param_groups = [{'params': params_wd, 'weight_decay': WEIGHT_DECAY},
                    {'params': params_rest}]
    optimizer = optim.SGD(param_groups, lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STETSIZE, gamma=scheduler_gama)

    loss_history = []
    acc_history = []
    counter = []
    iteration_number = 0
    diagram_data = DiagramData(loss_history, acc_history, counter, iteration_number)
    for epoch in range(1, epochs + 1):
        # scheduler.step()
        model = train_implement(model, device, train_loader, optimizer, epoch, target_steganography, diagram_data)
        if epoch % EVAL_PRINT_FREQUENCY == 0 or epoch == EPOCHS:
            print('Evaluate in epoch {}'.format(epoch))
            evaluate(model, device, valid_loader)
        print('current lr: ', optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()

    all_state = {
        'original_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    torch.save(all_state, params_save_file_path)
    torch.save(model, target_model_save_path)

    return model


def generate_model_save_file_name(dataset_enum, steganography_enum):
    model_save_file_name = 'SRNET_model_' + dataset_enum.name + '_' + steganography_enum.name + '04.pth'
    return model_save_file_name


def generate_model(device, target_dataset, target_steganography, reuse_model, reused_steganography=None,
                   reused_dataset=None):
    """
    :param device:
    :param target_dataset:
    :param target_steganography:
    :param reuse_model: 是否在之前已保存的模型基础上继续训练
    :param reused_steganography: 之前已保存的模型所使用的隐写方法。如果 @reuse_model 参数是 False，此参数可以不传。如果 @reuse_model 参数是 True，
    此参数又为空，默认赋值为 @target_steganography 的值
    :param reused_dataset: 之前已保存的模型所使用的数据集。如果 @reuse_model 参数是 False，此参数可以不传。
    :return:
    """
    model = SRNet().to(device)
    model.apply(init_weights)
    # 加载复用之前已保存的训练完的模型
    if reuse_model:
        if reused_steganography is None:
            print("选择了希望复用模型，但没有指定具体的复用模型，将使用目标隐写算法: {}".format(target_steganography.name))
            reused_steganography = target_steganography
        if reused_dataset is None:
            print('选择了希望复用模型，但没有指定复用模型的数据集，将使用目标数据集: {}'.format(target_dataset.name))
            reused_dataset = target_dataset
        reused_model_save_file_name = generate_model_save_file_name(reused_dataset, reused_steganography)
        reused_model_save_path = os.path.join(MODEL_EXPORT_PATH, reused_model_save_file_name)
        if not os.path.exists(reused_model_save_path):
            # raise RuntimeError('model_save_path 不存在')
            print('reused_model_save_path: {} 不存在'.format(reused_model_save_path))
        else:
            print('加载复用之前已保存的训练完的模型: {}'.format(reused_model_save_path))
            model = torch.load(reused_model_save_path, map_location=device)
    return model


def transfer_learning(dataset_steganography_list):
    init_console_log()
    device = torch.device("cuda" if use_gpu else "cpu")
    # 迁移学习训练
    train_dset_loaders = []
    valid_dset_loaders = []
    test_dset_loaders = []
    result_record = []
    for index, dataset_steganography in enumerate(dataset_steganography_list):
        dataset_enum = dataset_steganography['dataset']
        steganography_enum = dataset_steganography['steganography']
        test_loader, train_loader, valid_loader = generate_data_loaders(dataset_enum, steganography_enum)
        train_dset_loaders.append(train_loader)
        valid_dset_loaders.append(valid_loader)
        test_dset_loaders.append(test_loader)
    model = None
    for index, dataset_steganography in enumerate(dataset_steganography_list):
        pre_dataset = None
        pre_steganography = None
        dataset_enum = dataset_steganography['dataset']
        steganography_enum = dataset_steganography['steganography']
        if index > 0:
            pre_dataset_steganography = dataset_steganography_list[index - 1]
            pre_dataset = pre_dataset_steganography['dataset']
            pre_steganography = pre_dataset_steganography['steganography']
        dataloader_train = train_dset_loaders[index]
        dataloader_valid = valid_dset_loaders[index]
        dataloader_test = test_dset_loaders[index]
        if index == 0:
            model = generate_model(device, dataset_enum, steganography_enum, False)
            # model = generate_model(device, dataset_enum, steganography_enum, True)
        # else:
        #     model = generate_model(device, dataset_enum, steganography_enum, True, pre_steganography, pre_dataset)
        model = train_model(model, dataset_enum, steganography_enum, dataloader_train, dataloader_valid, EPOCHS, LR)
        print('\nevaluate model in {}'.format(steganography_enum.name))
        accuracy, test_loss = evaluate(model, device, dataloader_test)

        result_record.append(('[' + dataset_enum.name + '-' + steganography_enum.name + ']', accuracy, test_loss))

    # 释放显存
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    # 迁移学习结束之后用最后得到的模型回头测试前面的任务表现
    last_dataset_steganography = dataset_steganography_list[-1]
    last_dataset = last_dataset_steganography['dataset']
    last_steganography = last_dataset_steganography['steganography']
    model = generate_model(device, None, None, True, last_steganography, last_dataset)
    for index, dataset_steganography in enumerate(dataset_steganography_list):
        dataset_enum = dataset_steganography['dataset']
        steganography_enum = dataset_steganography['steganography']
        dataloader_test = test_dset_loaders[index]
        print('Test transfer learning {}-{}model\'s performance in former steganography {}-{}'
              .format(last_dataset.name, last_steganography.name, dataset_enum.name, steganography_enum.name))
        evaluate(model, device, dataloader_test)
        record_name, accuracy, test_loss = result_record[index]
        print('in former special training {} model\'s performance is accuracy: {}, test_loss: {}'
              .format(record_name, accuracy, test_loss))


def init_console_log():
    log_file_name = 'ste_transfer_learning'
    for ste in ste_list:
        dataset_name = ste['dataset'].name
        steganography_name = ste['steganography'].name
        log_file_name = log_file_name + '[' + dataset_name + '-' + steganography_name + ']'
    log_file_name = log_file_name + ',num_epochs-{}'.format(EPOCHS)
    log_file_name = log_file_name + '.log'
    log_file = os.path.join(LOG_PATH, log_file_name)
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    common_utils.ConsoleLogger(log_file)


class DiagramData:
    def __init__(self, loss_history, acc_history, counter, iteration_number):
        self.loss_history = loss_history
        self.acc_history = acc_history
        self.counter = counter
        self.iteration_number = iteration_number


def init_logger(dataset_enum, steganography_enum):
    # Log files
    # log_name = 'SRNET_params_boss_256_HILL04.log'
    log_name = 'SRNET_params_' + dataset_enum.name + '_' + steganography_enum.name + '04.log'
    log_path = os.path.join(LOG_PATH, log_name)
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    common_utils.set_logger(log_path, mode='w')


def generate_data_loaders(dataset_enum, steganography_enum):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_transform = transforms.Compose([
        steganalysis_utils.AugData(),
        steganalysis_utils.ToTensor()
    ])
    eval_transform = transforms.Compose([
        steganalysis_utils.ToTensor()
    ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = MyDataset(dataset_enum=dataset_enum, steganography_enum=steganography_enum, transform=transform)
    test_size_ratio = 0.2
    valid_size_ratio = 0.2
    dataset_size = len(dataset)
    indices_data = list(range(dataset_size))
    np.random.shuffle(indices_data)
    split_tt = int(np.floor(test_size_ratio * dataset_size))
    train_idx, test_idx = indices_data[split_tt:], indices_data[:split_tt]
    # For Valid
    num_train = len(train_idx)
    indices_train = list(range(num_train))
    np.random.shuffle(indices_train)
    split_tv = int(np.floor(valid_size_ratio * num_train))
    train_new_idx, valid_idx = indices_train[split_tv:], indices_train[:split_tv]
    train_sampler = SubsetRandomSampler(train_new_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, )
    valid_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=valid_sampler, )
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler, )
    return test_loader, train_loader, valid_loader


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # individual_learn(SteganographyEnum.HILL, False, SteganographyEnum.HILL)
    # transfer_learning([SteganographyEnum.HILL, SteganographyEnum.SUNI, SteganographyEnum.UTGAN])
    ste_list = [{'dataset': DatasetEnum.BOSSBase_256, 'steganography': SteganographyEnum.WOW},
                {'dataset': DatasetEnum.BOWS2_256, 'steganography': SteganographyEnum.SUNI},
                {'dataset': DatasetEnum.BOSSBase_256, 'steganography': SteganographyEnum.UTGAN},
                {'dataset': DatasetEnum.BOWS2_256, 'steganography': SteganographyEnum.HILL},
                ]
    transfer_learning(ste_list)

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import copy
import logging

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
from MAS import mas
from MAS.masUtils import utils, model_utils

BATCH_SIZE = 32
EPOCHS = 100
LR = 0.01
WEIGHT_DECAY = 5e-4
TRAIN_PRINT_FREQUENCY = 50
EVAL_PRINT_FREQUENCY = 1
STETSIZE = 14
scheduler_gama = 0.40
# DECAY_EPOCH = [30, 60, 90, 140, 200, 250, 300, 350]
# DECAY_EPOCH = [20, 60, 90, 120, 150, 170, 190]
DECAY_EPOCH = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
LOG_PATH = 'data/log'
DATASET_DIR = r'D:\Work\dataset\steganalysis\BOSSBase'
use_gpu = torch.cuda.is_available()
reuse_model = True

train_dset_loaders = []
valid_dset_loaders = []
test_dset_loaders = []


class SteganographyEnum(Enum):
    HILL = 1
    SUNI = 2
    UTGAN = 3


def evaluate(model, device, data_loader, epoch):
    logging.info('start evaluate')
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
            # data = data[idx]
            # label = label[idx]

            data, label = data.to(device), label.to(device)
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, label)
            test_loss += loss.item()
            batch_num += 1

            pred = output.max(1, keepdim=True)[1]
            total += label.size(0)
            correct += pred.eq(label.view_as(pred)).sum().item()

    # accuracy = correct / (len(eval_loader.dataset) * 2)
    accuracy = 100. * correct / total

    if accuracy > best_acc:
        best_acc = accuracy
        # all_state = {
        #     'original_state': model.state_dict(),
        #     'optimizer_state': optimizer.state_dict(),
        #     'epoch': epoch
        # }
        # torch.save(all_state, params_path)

    logging.info('-' * 8)
    test_loss = test_loss / batch_num
    logging.info('Test in epoch {}'.format(epoch))
    logging.info('Test loss: {:.4f}'.format(test_loss))
    logging.info('Eval accuracy: {:.4f}'.format(accuracy))
    logging.info('Best accuracy:{:.4f}'.format(best_acc))
    logging.info('-' * 8)
    return best_acc, test_loss


def set_logger(log_path, mode='a'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode=mode)
        file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def train_implement(model, device, train_loader, optimizer, epoch):
    # losses = steganalysis_utils.AverageMeter()
    model.train()
    train_loss = 0
    train_correct = 0
    total = 0

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
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)
        # losses.update(loss.item(), data.size(0))
        loss.backward()  # BP
        optimizer.step()
        train_loss += loss.item()
        prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
        total += label.size(0)
        train_correct += np.sum(prediction[1].cpu().numpy() == label.cpu().numpy())

        if i % TRAIN_PRINT_FREQUENCY == 0:
            logging.info('train epoch: [{0}][{1}/{2}]\t'
                         'Acc {acc:.4f}\t'
                         'Loss {loss:.4f} \t'
                         .format(epoch, i, len(train_loader), acc=100. * train_correct / total,
                                 loss=train_loss / (i + 1)))
    return model


def train_model(device, model, params_save_file_path, train_loader, valid_loader):
    params = model.parameters()
    params_wd, params_rest = [], []
    for param_item in params:
        if param_item.requires_grad:
            (params_wd if param_item.dim() != 1 else params_rest).append(param_item)
    param_groups = [{'params': params_wd, 'weight_decay': WEIGHT_DECAY},
                    {'params': params_rest}]
    optimizer = optim.SGD(param_groups, lr=LR, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STETSIZE, gamma=scheduler_gama)

    for epoch in range(1, EPOCHS + 1):
        # scheduler.step()
        model = train_implement(model, device, train_loader, optimizer, epoch)
        if epoch % EVAL_PRINT_FREQUENCY == 0 or epoch == EPOCHS:
            evaluate(model, device, valid_loader, epoch)
        print('current lr: ', optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()

    all_state = {
        'original_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    torch.save(all_state, params_save_file_path)
    return model


def init_logger(steganography_enum):
    # Log files
    # log_name = 'SRNET_params_boss_256_HILL04.log'
    log_name = 'SRNET_params_boss_256_' + steganography_enum.name + '04.log'
    log_path = os.path.join(LOG_PATH, log_name)
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    set_logger(log_path, mode='w')


def generate_data_loaders(steganography_enum):
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

    dataset = MyDataset(dataset_dir=DATASET_DIR, steganography_enum=steganography_enum, transform=transform)
    test_size_ratio = 0.2
    valid_size_ratio = 0.1
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


num_epochs = 50
num_freeze_layers = 0
lr = 0.001
reg_lambda = 1


def main(steganography_enums):
    device = torch.device("cuda" if use_gpu else "cpu")

    for steganography_enum in steganography_enums:
        init_logger(steganography_enum)
        test_loader, train_loader, valid_loader = generate_data_loaders(steganography_enum)
        train_dset_loaders.append(train_loader)
        valid_dset_loaders.append(valid_loader)
        test_dset_loaders.append(test_loader)

    # get the number of tasks in the sequence
    no_of_tasks = len(train_dset_loaders)
    # train the model on the given number of tasks
    for task in range(1, no_of_tasks + 1):
        print("Training the model on task {}".format(task))

        dataloader_train = train_dset_loaders[task - 1]
        dataloader_valid = valid_dset_loaders[task - 1]

        # no_of_classes = dataloader_train.dataset.classes
        no_of_classes = 2

        model = model_utils.model_init(no_of_classes, use_gpu)

        mas.mas_train(model, task, num_epochs, num_freeze_layers, no_of_classes, dataloader_train, dataloader_valid, lr,
                      reg_lambda, use_gpu)

    print("The training process on the {} tasks is completed".format(no_of_tasks))

    print("Testing the model now")

    # test the model out on the test sets of the tasks
    for task in range(1, no_of_tasks + 1):
        print("Testing the model on task {}".format(task))

        dataloader_test = test_dset_loaders[task - 1]
        # no_of_classes = dataloader_test.dataset.classes
        no_of_classes = 2

        # load the model for inference
        model = model_utils.model_inference(task, use_gpu)
        model.to(device)

        forgetting = mas.compute_forgetting(model, task, dataloader_test, use_gpu)

        print("The forgetting undergone on task {} is {:.4f}".format(task, forgetting))


if __name__ == '__main__':
    main([SteganographyEnum.HILL, SteganographyEnum.SUNI, SteganographyEnum.UTGAN])

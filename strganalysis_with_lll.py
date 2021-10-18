# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
# import pandas as pd
# from pathlib import Path
# import copy
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
from common import DatasetEnum, SteganographyEnum
import common_utils

BATCH_SIZE = 32
# DECAY_EPOCH = [30, 60, 90, 140, 200, 250, 300, 350]
# DECAY_EPOCH = [20, 60, 90, 120, 150, 170, 190]
DECAY_EPOCH = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
LOG_PATH = 'data/log'
DATASET_DIR = r'D:\Work\dataset\steganalysis\BOSSBase'
use_gpu = torch.cuda.is_available()

num_epochs = 60
num_freeze_layers = 0
lr = 0.01
reg_lambda = 1

train_dset_loaders = []
valid_dset_loaders = []
test_dset_loaders = []


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


def init_logger(dataset_enum, steganography_enum):
    # Log files
    # log_name = 'SRNET_params_boss_256_HILL04.log'
    log_name = 'SRNET_params_' + dataset_enum.name + '_' + steganography_enum.name + '04.log'
    log_path = os.path.join(LOG_PATH, log_name)
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    set_logger(log_path, mode='w')


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


def main(dataset_steganography_list, reuse_model):
    init_console_log()
    device = torch.device("cuda" if use_gpu else "cpu")

    for dataset_steganography in dataset_steganography_list:
        dataset_enum = dataset_steganography['dataset']
        steganography_enum = dataset_steganography['steganography']
        init_logger(dataset_enum, steganography_enum)
        test_loader, train_loader, valid_loader = generate_data_loaders(dataset_enum, steganography_enum)
        train_dset_loaders.append(train_loader)
        valid_dset_loaders.append(valid_loader)
        test_dset_loaders.append(test_loader)

    # get the number of tasks in the sequence
    no_of_tasks = len(train_dset_loaders)
    # train the model on the given number of tasks
    for task in range(1, no_of_tasks + 1):
        dataloader_train = train_dset_loaders[task - 1]
        dataloader_valid = valid_dset_loaders[task - 1]

        # no_of_classes = dataloader_train.dataset.classes
        no_of_classes = 2

        model = model_utils.model_init(task, no_of_classes, use_gpu, reuse_model)
        # 从第二个任务开始，初始的lr每次缩小10倍
        global lr
        lr = lr * pow(0.1, task - 1)
        print("Training the model on task {}, λ = {}, lr = {}".format(task, reg_lambda, lr))

        mas.mas_train(model, task, num_epochs, num_freeze_layers, no_of_classes, dataloader_train, dataloader_valid, lr,
                      reg_lambda, use_gpu)

    print("The training process on the {} tasks is completed".format(no_of_tasks))

    print("Testing the model now")
    # 释放显存
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    # test the model out on the test sets of the tasks
    for task in range(1, no_of_tasks + 1):
        print("Testing the model on task {}".format(task))

        dataloader_test = test_dset_loaders[task - 1]
        # no_of_classes = dataloader_test.dataset.classes
        no_of_classes = 2

        # load the model for inference
        model = model_utils.model_inference(task, use_gpu)
        model.to(device)
        print("model: ", model)

        forgetting = mas.compute_forgetting(model, task, dataloader_test, use_gpu)

        print("The forgetting undergone on task {} is {:.4f}".format(task, forgetting))


def init_console_log():
    log_file_name = 'str_with_lll_'
    for ste in ste_list:
        dataset_name = ste['dataset'].name
        steganography_name = ste['steganography'].name
        log_file_name = log_file_name + '[' + dataset_name + '-' + steganography_name + ']'
    log_file_name = log_file_name + ',num_epochs-{},reg_lambda-{}'.format(num_epochs, reg_lambda)
    log_file_name = log_file_name + '.log'
    log_file = os.path.join(LOG_PATH, log_file_name)
    if not os.path.exists(log_file):
        os.makedirs(log_file)
    common_utils.Logger(log_file)


if __name__ == '__main__':
    # main([SteganographyEnum.HILL, SteganographyEnum.SUNI, SteganographyEnum.UTGAN], False)
    ste_list = [{'dataset': DatasetEnum.BOSSBase_256, 'steganography': SteganographyEnum.HILL},
                {'dataset': DatasetEnum.BOWS2OrigEp3, 'steganography': SteganographyEnum.SUNI},
                {'dataset': DatasetEnum.BOSSBase_256, 'steganography': SteganographyEnum.UTGAN}]
    main(ste_list, False)

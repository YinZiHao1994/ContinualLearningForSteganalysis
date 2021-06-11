# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import copy
import logging
import random
import scipy.io as sio
import matplotlib.pyplot as plt
import time
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
# 30 SRM filtes
from srm_filter_kernel import all_normalized_hpf_list
# Global covariance pooling
from MPNCOV import *  # MPNCOV

BATCH_SIZE = 32
EPOCHS = 100
LR = 0.01
WEIGHT_DECAY = 5e-4
TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1
STETSIZE = 14
scheduler_gama = 0.40
# DECAY_EPOCH = [30, 60, 90, 140, 200, 250, 300, 350]
# DECAY_EPOCH = [20, 60, 90, 120, 150, 170, 190]
DECAY_EPOCH = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
OUTPUT_PATH = '/data/cuiqi/steganalysis_with_CNN_Yedroudj-Net-master/pytorch_version/log'




class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, device, train_loader, optimizer, epoch):
    losses = AverageMeter()
    model.train()
    train_loss = 0
    train_correct = 0
    total = 0

    for i, sample in enumerate(train_loader):

        data, label = sample['data'], sample['label']
        shape = list(data.size())
        data = data.reshape(shape[0] * shape[1], *shape[2:])
        label = label.reshape(-1)
        ##shuffle##
        idx = torch.randperm(shape[0])
        data = data[idx]
        label = label[idx]

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
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Acc {acc:.4f}\t'
                         'Loss {loss:.4f} \t'.format(
                epoch, i, len(train_loader), acc=100. * train_correct / total, loss=train_loss / (i + 1)))


def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0
    batch_num = 0

    with torch.no_grad():
        for i, sample in enumerate(eval_loader):
            data, label = sample['data'], sample['label']
            shape = list(data.size())
            data = data.reshape(shape[0] * shape[1], *shape[2:])
            label = label.reshape(-1)
            ##shuffle##
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

    if accuracy > best_acc and epoch > 40:
        best_acc = accuracy
        all_state = {
            'original_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(all_state, PARAMS_PATH)

    logging.info('-' * 8)
    logging.info('Test loss: {:.4f}'.format(test_loss / batch_num))
    logging.info('Eval accuracy: {:.4f}'.format(accuracy))
    logging.info('Best accuracy:{:.4f}'.format(best_acc))
    logging.info('-' * 8)
    return best_acc, test_loss / batch_num


# Initialization
def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
        nn.init.constant_(module.bias.data, val=0)


# Data augmentation
class AugData():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        # Rotation
        rot = random.randint(0, 3)
        data = np.rot90(data, rot, axes=[1, 2]).copy()

        # Mirroring
        if random.random() < 0.5:
            data = np.flip(data, axis=2).copy()

        new_sample = {'data': data, 'label': label}

        return new_sample


class ToTensor():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        data = np.expand_dims(data, axis=1)
        data = data.astype(np.float32)
        # data = data / 255.0

        new_sample = {
            'data': torch.from_numpy(data),
            'label': torch.from_numpy(label).long(),
        }

        return new_sample


'''
class MyDataset(Dataset):
    def __init__(self, DATASET_DIR, partition, transform=None):
        random.seed(1234)

        self.transform = transform

        self.cover_dir = DATASET_DIR + '/cover'
        self.stego_dir = DATASET_DIR + '/stego/' + Model_NAME

        self.covers_list = [x.split('/')[-1] for x in glob(self.cover_dir + '/*')]
        random.shuffle(self.covers_list)
        if (partition == 0):
            self.cover_list = self.covers_list[:4000]
        if (partition == 1):
            self.cover_list = self.covers_list[4000:5000]
        if (partition == 2):
            self.cover_list = self.covers_list[5000:10000]
        assert len(self.covers_list) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.cover_list)

    def __getitem__(self, idx):
        file_index = int(idx)

        cover_path = os.path.join(self.cover_dir, self.cover_list[file_index])
        stego_path = os.path.join(self.stego_dir, self.cover_list[file_index])

        cover_data = cv2.imread(cover_path, -1)
        stego_data = cv2.imread(stego_path, -1)

        data = np.stack([cover_data, stego_data])
        label = np.array([0, 1], dtype='int32')

        sample = {'data': data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
'''


class MyDataset(Dataset):
    def __init__(self, DATASET_DIR, transform=None):
        self.transform = transform

        self.cover_dir = DATASET_DIR + '/BOSSBase_256'
        # self.stego_dir = DATASET_DIR + '/stego_suniward04'
        self.stego_dir = DATASET_DIR + '/BOSSBase_256_HILL04'

        self.cover_list = [x.split('/')[-1] for x in glob(self.cover_dir + '/*')]
        assert len(self.cover_list) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.cover_list)

    def __getitem__(self, idx):
        file_index = int(idx)

        cover_path = os.path.join(self.cover_dir, self.cover_list[file_index])
        stego_path = os.path.join(self.stego_dir, self.cover_list[file_index])

        # cover_data = cv2.imread(cover_path, 0)
        # stego_data = cv2.imread(stego_path, 0)
        cover_data = Image.open(cover_path)  # .convert('RGB')
        stego_data = Image.open(stego_path)  # .convert('RGB')
        # cover_data = np.array(cover_data)
        # stego_data = np.array(stego_data)

        # data = np.stack([cover_data, stego_data], ).transpose((0, 3, 1, 2))
        # print(np.array(cover_data).shape)
        # data_ = Image.fromarray(data).convert('RGB')
        label = np.array([0, 1], dtype='uint8')
        label = torch.from_numpy(label).long()
        # print(type(data) )
        # print(type(label) )

        # sample = {'data': data, 'label': label}
        # print(type(Image.fromarray(data)) )

        if self.transform:
            cover_data = self.transform(cover_data)
            stego_data = self.transform(stego_data)
            # print('cover_data',cover_data.shape)
        # data = torch.cat((cover_data, stego_data), 0)
        data = torch.stack((cover_data, stego_data))
        # print('data', data.shape)

        sample = {'data': data, 'label': label}
        return sample


def setLogger(log_path, mode='a'):
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


def main():
    device = torch.device("cuda")
    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_transform = transforms.Compose([
        AugData(),
        ToTensor()
    ])

    eval_transform = transforms.Compose([
        ToTensor()
    ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Log files
    PARAMS_NAME = 'model_params_boss_HILL04_256_SRNET.pt'
    LOG_NAME = 'model_params_boss_HILL04_256_SRNET.log'

    PARAMS_PATH = os.path.join(OUTPUT_PATH, PARAMS_NAME)
    LOG_PATH = os.path.join(OUTPUT_PATH, LOG_NAME)

    setLogger(LOG_PATH, mode='w')

    # Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    data = MyDataset(DATASET_DIR='/data/BossClf256', transform=transform)
    test_size = 0.2
    valid_size = 0.1
    num_data = len(data)
    indices_data = list(range(num_data))

    np.random.shuffle(indices_data)

    split_tt = int(np.floor(test_size * num_data))
    train_idx, test_idx = indices_data[split_tt:], indices_data[:split_tt]
    # For Valid
    num_train = len(train_idx)
    indices_train = list(range(num_train))
    np.random.shuffle(indices_train)
    split_tv = int(np.floor(valid_size * num_train))
    train_new_idx, valid_idx = indices_train[split_tv:], indices_train[:split_tv]

    train_sampler = SubsetRandomSampler(train_new_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(data, batch_size=BATCH_SIZE, sampler=train_sampler, )
    valid_loader = DataLoader(data, batch_size=BATCH_SIZE, sampler=valid_sampler, )
    test_loader = DataLoader(data, batch_size=BATCH_SIZE, sampler=test_sampler, )

    model = SRNet().to(device)
    model.apply(initWeights)
    params = model.parameters()

    params_wd, params_rest = [], []
    for param_item in params:
        if param_item.requires_grad:
            (params_wd if param_item.dim() != 1 else params_rest).append(param_item)

    param_groups = [{'params': params_wd, 'weight_decay': WEIGHT_DECAY},
                    {'params': params_rest}]

    optimizer = optim.SGD(param_groups, lr=LR, momentum=0.9, weight_decay=0.0005)

    startEpoch = 1
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=scheduler_gama)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STETSIZE, gamma=scheduler_gama)
    best_acc = 0.0

    for epoch in range(startEpoch, EPOCHS + 1):
        # scheduler.step()
        train(model, device, train_loader, optimizer, epoch)
        if epoch % EVAL_PRINT_FREQUENCY == 0:
            best_acc, test_loss = evaluate(model, device, valid_loader, epoch, optimizer, best_acc, PARAMS_PATH)
        print('current lr: ', optimizer.state_dict()['param_groups'][0]['lr'])
        # scheduler.step(test_loss)
        scheduler.step()
    logging.info('\nTest set accuracy: \n')

    # Load best network parmater to test
    all_state = torch.load(PARAMS_PATH)
    original_state = all_state['original_state']
    optimizer_state = all_state['optimizer_state']
    model.load_state_dict(original_state)
    optimizer.load_state_dict(optimizer_state)

    evaluate(model, device, test_loader, epoch, optimizer, best_acc, PARAMS_PATH)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()

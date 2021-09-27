#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import numpy as np
from random import shuffle
import copy
import sys

from masUtils.model_utils import *
from masUtils.utils import *
from model_class import *
from optimizer_lib import *
from model_train import *
from mas import *

# sys.path.append('./masUtils')
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Test file')
parser.add_argument('--use_gpu', default=False, type=bool, help='Set the flag if you wish to use the GPU')
parser.add_argument('--batch_size', default=32, type=int, help='The batch size you want to use')
parser.add_argument('--num_freeze_layers', default=0, type=int,
                    help='Number of layers you want to frozen in the feature extractor of the model')
parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs you want to train the model on')
parser.add_argument('--init_lr', default=0.001, type=float, help='Initial learning rate for training the model')
parser.add_argument('--reg_lambda', default=1, type=float, help='Regularization parameter')

args = parser.parse_args()
# use_gpu = args.use_gpu
use_gpu = torch.cuda.is_available()
batch_size = args.batch_size
num_freeze_layers = args.num_freeze_layers
num_epochs = args.num_epochs
lr = args.init_lr
reg_lambda = args.reg_lambda

train_dset_loaders = []
test_dset_loaders = []

train_dsets_size = []
test_dsets_size = []

num_classes = []

data_path = os.path.join(os.getcwd(), "Data")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = os.path.join(os.getcwd(), "Data")

# create the dataloaders for all the tasks
for task_dir in sorted(os.listdir(data_dir)):
    # create the image folders objects
    tr_image_folder = datasets.ImageFolder(os.path.join(data_dir, task_dir, "train"),
                                           transform=data_transforms['train'])
    te_image_folder = datasets.ImageFolder(os.path.join(data_dir, task_dir, "test"), transform=data_transforms['test'])

    # get the dataloaders
    # tr_dset_loaders = torch.masUtils.data.DataLoader(tr_image_folder, batch_size=batch_size, shuffle=True, num_workers=4)
    tr_dset_loader = torch.utils.data.DataLoader(tr_image_folder, batch_size=batch_size, shuffle=True)
    # te_dset_loaders = torch.masUtils.data.DataLoader(te_image_folder, batch_size=batch_size, shuffle=True, num_workers=4)
    te_dset_loader = torch.utils.data.DataLoader(te_image_folder, batch_size=batch_size, shuffle=True)

    # get the sizes
    train_len = len(tr_image_folder)
    test_len = len(te_image_folder)

    # append the dataloaders of these tasks
    train_dset_loaders.append(tr_dset_loader)
    test_dset_loaders.append(te_dset_loader)

    # get the classes (THIS MIGHT NEED TO BE CORRECTED)
    num_classes.append(len(tr_image_folder.classes))

    # get the sizes array
    train_dsets_size.append(train_len)
    test_dsets_size.append(test_len)

# get the number of tasks in the sequence
no_of_tasks = len(train_dset_loaders)

# train the model on the given number of tasks
for task in range(1, no_of_tasks + 1):
    print("Training the model on task {}".format(task))

    dataloader_train = train_dset_loaders[task - 1]
    dataloader_test = test_dset_loaders[task - 1]
    dset_size_train = train_dsets_size[task - 1]
    dset_size_test = test_dsets_size[task - 1]

    no_of_classes = num_classes[task - 1]

    model = model_init(task, no_of_classes, use_gpu)

    mas_train(model, task, num_epochs, num_freeze_layers, no_of_classes, dataloader_train, dataloader_test, lr,
              reg_lambda, use_gpu)

print("The training process on the {} tasks is completed".format(no_of_tasks))

print("Testing the model now")

# test the model out on the test sets of the tasks
for task in range(1, no_of_tasks + 1):
    print("Testing the model on task {}".format(task))

    dataloader = test_dset_loaders[task - 1]
    dset_size = test_dsets_size[task - 1]
    no_of_classes = num_classes[task - 1]

    # load the model for inference
    model = model_inference(task, use_gpu)
    device = torch.device("cuda:0" if use_gpu else "cpu")
    model.to(device)

    forgetting = compute_forgetting(model, task, dataloader, use_gpu)

    print("The forgetting undergone on task {} is {:.4f}".format(task, forgetting))

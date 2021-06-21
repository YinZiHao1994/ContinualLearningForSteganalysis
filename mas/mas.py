#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import copy
import os
import shutil

import sys
import time

# sys.path.append('./utils')
from utils.model_utils import *
from utils.mas_utils import *

from optimizer_lib import *
from model_train import *


def mas_train(model, task_no, num_epochs, num_freeze_layers, no_of_classes, dataloader_train, dataloader_test, lr=0.001,
              reg_lambda=0.01, use_gpu=False):
    """
    Inputs:
    1) model: A reference to the model that is being exposed to the data for the task
    2) task_no: The task that is being exposed to the model identified by it's number
    3) num_freeze_layers: The number of layers that you want to freeze in the feature extractor of the Alexnet
    4) no_of_classes: The number of classes in the task
    5) dataloader_train: Dataloader that feeds training data to the model
    6) dataloader_test: Dataloader that feeds test data to the model
    6) dset_size_train: The size of the task (size of the dataset belonging to the training task)
    7) dset_size_test: The size of the task (size of the dataset belonging to the test set)
    7) use_gpu: Set the flag to `True` if you want to train the model on GPU

    Outputs:
    1) model: Returns a trained model

    Function: Trains the model on a particular task and deals with different tasks in the sequence
    """

    # this is the task to which the model is exposed
    if task_no == 1:
        # initialize the reg_params for this task
        model, freeze_layers = create_freeze_layers(model, num_freeze_layers)
        model = init_reg_params(model, use_gpu, freeze_layers)

    else:
        # initialize the reg_params for this task
        model = init_reg_params_across_tasks(model, use_gpu)

    # get the optimizer
    optimizer_sp = LocalSgd(model.tmodel.parameters(), reg_lambda, lr)
    train_model(model, task_no, no_of_classes, optimizer_sp, model_criterion, dataloader_train, dataloader_test,
                num_epochs, use_gpu, lr, reg_lambda)

    if task_no > 1:
        model = consolidate_reg_params(model, use_gpu)

    return model


def compute_forgetting(model, task_no, dataloader, use_gpu):
    """
    Inputs
    1) task_no: The task number on which you want to compute the forgetting
    2) dataloader: The dataloader that feeds in the sample to the model

    Outputs
    1) forgetting: The amount of forgetting undergone by the model

    Function: Computes the "forgetting" that the model has on the
    """

    # get the results file
    store_path = os.path.join(os.getcwd(), "models", "Task_" + str(task_no))
    model_path = os.path.join(os.getcwd(), "models")
    device = torch.device("cuda:0" if use_gpu else "cpu")

    # get the old performance
    file_object = open(os.path.join(store_path, "performance.txt"), 'r')
    old_performance = file_object.read()
    file_object.close()

    running_corrects = 0.0

    for sample in dataloader:
        datas, labels = sample['data'], sample['label']
        shape = list(datas.size())
        datas = datas.reshape(shape[0] * shape[1], *shape[2:])
        labels = labels.reshape(-1)
        # shuffle
        idx = torch.randperm(shape[0])
        data = datas[idx]
        label = labels[idx]
        del sample

        if use_gpu:
            data = data.to(device)
            label = label.to(device)

        else:
            data = Variable(data)
            label = Variable(label)

        output = model.tmodel(data)
        del data

        # running_corrects += torch.sum(preds == labels.data)
        prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced

        running_corrects += np.sum(prediction[1].cpu().numpy() == label.cpu().numpy())
        del labels

    dset_size = len(dataloader.dataset)
    epoch_accuracy = running_corrects / dset_size

    old_performance = float(old_performance)
    forgetting = epoch_accuracy.item() - old_performance
    print("on task {} the old_accuracy and new_accuracy  is {} and {}".format(task_no, old_performance, epoch_accuracy))
    return forgetting

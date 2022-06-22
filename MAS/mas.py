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
# from MAS.masUtils.model_utils import *
from MAS.masUtils.model_utils import *
# from mas_utils import *

from MAS.optimizer_lib import *
from MAS.model_train import *


def mas_train(model, task_no, num_epochs, num_freeze_layers, no_of_classes, dataloader_train, dataloader_valid, lr=0.001,
              reg_lambda=0.01, use_awl=False, use_gpu=False):
    """
    Outputs:
    1) model: Returns a trained model

    Function: Trains the model on a particular task and deals with different tasks in the sequence
    :param model: A reference to the model that is being exposed to the data for the task
    :param task_no: The task that is being exposed to the model identified by it's number
    :param num_freeze_layers: The number of layers that you want to freeze in the feature extractor of the Alexnet
    :param no_of_classes: The number of classes in the task
    :param dataloader_train: Dataloader that feeds training data to the model
    :param dataloader_valid: Dataloader that feeds test data to the model
    :param dset_size_train: The size of the task (size of the dataset belonging to the training task)
    :param dset_size_test: The size of the task (size of the dataset belonging to the test set)
    :param use_gpu: Set the flag to `True` if you want to train the model on GPU
    :param use_awl:
    """

    # this is the task to which the model is exposed
    if task_no == 1:
        # initialize the reg_params for this task
        model, freeze_layers = create_freeze_layers(model, num_freeze_layers)
        model = init_reg_params(model, use_gpu, model.lambda_list, freeze_layers)

    else:
        # initialize the reg_params for this task
        model = init_reg_params_across_tasks(model, task_no, use_gpu)

    # if task_no > 1:
    #     model = consolidate_reg_params(model, use_gpu)
    # optimizer_sp = LocalSgd(model.tmodel.parameters(), reg_lambda, lr, momentum=momentum, weight_decay=0.0005)
    # optimizer_sp = LocalSgd(filter(lambda p: p.requires_grad, model.tmodel.parameters()), reg_lambda,
    #                         model.weight_params, lr=lr, momentum=momentum, weight_decay=0.0005)
    # optimizer_sp = LocalSgd(
    #     filter(lambda p: (p.requires_grad is not None and p.requires_grad) or p.requires_grad is None,
    #            model.tmodel.parameters()), reg_lambda, model.weight_params, lr, momentum=momentum, weight_decay=0.0005)
    # optimizer_sp = optim.SGD(
    #     filter(lambda p: (p.requires_grad is not None and p.requires_grad) or p.requires_grad is None,
    #            model.tmodel.parameters()), lr, momentum=momentum, weight_decay=0.0005)

    train_model(model, task_no, no_of_classes, model_criterion, dataloader_train, dataloader_valid, num_epochs, use_gpu,
                lr, reg_lambda, use_awl)

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
    total = 0
    for index, sample in enumerate(dataloader):
        if index % 50 == 0:
            print("sample {}/{} in dataloader".format(index, len(dataloader)))
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
        total += label.size(0)

    dset_size = len(dataloader.dataset)
    epoch_accuracy = running_corrects / total

    old_performance = float(old_performance)
    epoch_accuracy = (epoch_accuracy.item() if torch.is_tensor(epoch_accuracy) else epoch_accuracy)
    forgetting = epoch_accuracy - old_performance
    print("on task {} the old_accuracy and new_accuracy  is {} and {}".format(task_no, old_performance, epoch_accuracy))
    return forgetting

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
import pickle

import sys

from MAS.model_class import *
from SRNet import SRNet

# sys.path.append('../')
MODEL_EXPORT_PATH = 'data/model_export'
PATH_TO_MODEL = os.path.join(os.getcwd(), "models")


def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=20):
    """
    Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.

    """
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    print('lr is ' + str(lr))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def model_criterion(preds, labels):
    """
    Function: Model criterion to train the model

    """
    loss = nn.CrossEntropyLoss()
    return loss(preds, labels)


def check_checkpoints(store_path):
    """
    Inputs
    1) store_path: The path where the checkpoint file will be searched at

    Outputs
    1) checkpoint_file: The checkpoint file if it exists
    2) flag: The flag will be set to True if the directory exists at the path

    Function: This function takes in the store_path and checks if a prior directory exists
    for the task already. If it doesn't, flag is set to False and the function returns an empty string.
    If a directory exists the function returns a checkpoint file

    """

    # if the directory does not exist return an empty string
    if not os.path.isdir(store_path):
        return ["", False]

    # directory exists but there is no checkpoint file
    onlyfiles = [f for f in os.listdir(store_path) if os.path.isfile(os.path.join(store_path, f))]
    max_train = -1
    flag = False
    checkpoint_file = ""
    # Check the latest epoch file that was created
    for file in onlyfiles:
        if file.endswith('pth.tr'):
            flag = True
            test_epoch = file[0]
            if test_epoch > max_train:
                max_epoch = test_epoch
                checkpoint_file = file

    # no checkpoint exists in the directory so return an empty string
    if not flag:
        checkpoint_file = ""

    return [checkpoint_file, True]


def create_task_dir(task_no, no_of_classes, store_path):
    """
    Inputs
    1) task_no: The identity for the task defined by it's number in the sequence
    2) no_of_classes: The number of classes that the particular task has


    Function: This function creates a directory to store the classification head for the new task. It also
    creates a text file which stores the number of classes that this task contained.

    """
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    file_path = os.path.join(store_path, "classes.txt")

    with open(file_path, 'w') as file1:
        input_to_txtfile = str(no_of_classes)
        file1.write(input_to_txtfile)
        file1.close()

    return


def model_inference(task_no, task_length, use_gpu=False):
    """
    Inputs
    1) task_no: The task number for which the model is being evaluated
    2) use_gpu: Set the flag to True if you want to run the code on GPU. Default value: False

    Outputs
    1) model: A reference to the model

    Function: Combines the classification head for a particular task with the shared model and
    returns a reference to the model is used for testing the process
    :param task_length:

    """
    device = torch.device("cuda" if use_gpu else "cpu")
    # pre_model = get_pre_model(use_gpu, False)
    # model = SharedModel(pre_model)

    path_to_model = os.path.join(os.getcwd(), "models")

    path_to_head = os.path.join(os.getcwd(), "models", "Task_" + str(task_no))

    # get the number of classes by reading from the text file created during initialization for this task
    file_name = os.path.join(path_to_head, "classes.txt")
    file_object = open(file_name, 'r')
    num_classes = file_object.read()
    file_object.close()

    num_classes = int(num_classes)
    # print (num_classes)
    # in_features = model.tmodel.classifier[-1].in_features

    complete_model_path = os.path.join(path_to_model, 'model_in_task_{}'.format(task_length) + '.pth.tar')
    if not os.path.isfile(complete_model_path):
        raise RuntimeError("保存的模型路径 {} 不存在".format(complete_model_path))
    print("load model from {}".format(complete_model_path))
    model = torch.load(complete_model_path)

    # 针对 SRNet 的情况
    # classifier = model.tmodel.fc
    # in_features = classifier.in_features
    #
    # # load the classifier head for the given task identified by the task number
    # classification_head = ClassificationHead(in_features, num_classes)
    # classification_head.load_state_dict(torch.load(os.path.join(path_to_head, "head.pth")))
    #
    # # load the trained shared model
    # # model.load_state_dict(torch.load(os.path.join(path_to_model, "shared_model.pth"), map_location=device))
    #
    # # model.tmodel.classifier.add_module('6', nn.Linear(in_features, num_classes))
    # model.tmodel.fc = nn.Linear(in_features, num_classes)
    #
    # # change the weights layers to the classifier head weights
    # model.tmodel.fc.weight.data = classification_head.fc.weight.data
    # model.tmodel.fc.bias.data = classification_head.fc.bias.data

    # device = torch.device("cuda:0" if use_gpu else "cpu")
    model.eval()
    # model.to(device)

    return model


# Initialization
def init_weights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
        nn.init.constant_(module.bias.data, val=0)


def get_pre_model(use_gpu=False, reuse_model=False):
    model_save_file_name = 'SRNET_model_boss_256_' + 'HILL' + '04.pth'
    model_save_path = os.path.join(MODEL_EXPORT_PATH, model_save_file_name)
    device = torch.device("cuda" if use_gpu else "cpu")
    model = SRNet().to(device)
    # 加载复用之前已保存的训练完的模型
    if reuse_model:
        if not os.path.exists(model_save_path):
            # raise RuntimeError('model_save_path 不存在')
            print('model_save_path: {} 不存在'.format(model_save_path))
            model.apply(init_weights)
        else:
            print('加载复用之前已保存的训练完的模型: {}'.format(model_save_path))
            model = torch.load(model_save_path, map_location=device)
    else:
        model.apply(init_weights)
    return model


def model_init(task_num, no_classes, use_gpu=False, reuse_model=True):
    """
    Inputs
    1) no_classes: The number of classes that the model is exposed to in the new task
    2) use_gpu: Set the flag to True if you want to run the code on GPU. Default value = False

    Outputs
    1) model: A reference to the model that has been initialized

    Function: Initializes a model for the new task which the shared features and a classification head
    particular to the new task
    :param task_num:
    :param no_classes:
    :param use_gpu:
    :param reuse_model:

    """
    path_to_model = os.path.join(os.getcwd(), "models")

    path = os.path.join(path_to_model, "shared_model.pth")
    path_to_reg = os.path.join(path_to_model, "reg_params.pickle")

    model = None
    if task_num == 1:
        pre_model = get_pre_model(use_gpu, reuse_model)
        model = SharedModel(pre_model)

        # # load the model
        # if os.path.isfile(path):
        #     print('加载 shared_model')
        #     model.load_state_dict(torch.load(path, map_location=device))
        #
        # # load the reg_params stored
        # if os.path.isfile(path_to_reg):
        #     with open(path_to_reg, 'rb') as handle:
        #         reg_params = pickle.load(handle)
        #
        #     model.reg_params = reg_params
    else:
        complete_model_path = os.path.join(path_to_model, 'model_in_task_{}'.format(task_num - 1) + '.pth.tar')
        if not os.path.isfile(complete_model_path):
            raise RuntimeError("保存的模型路径 {} 不存在".format(complete_model_path))
        print("load model from {}".format(complete_model_path))
        model = torch.load(complete_model_path)

    # 针对 隐写分析的 SRNet 的情况，所有任务都是二分类，直接使用默认的fc
    # initialize a new classification head
    # classifier = model.tmodel.fc
    # in_features = classifier.in_features
    # # add the last classfication head to the shared model
    # model.tmodel.fc = nn.Linear(in_features, no_classes)

    # 从第二个任务开始，冻结最后的全连接层
    # if task_num > 1:
    #     for par in model.tmodel.fc.parameters():
    #         par.requires_grad = False
    device = torch.device("cuda:0" if use_gpu else "cpu")
    model.train(True)
    model.to(device)

    return model


def save_model(model, task_no):
    """
    Inputs
    1) model: A reference to the model that needs to be saved
    2) task_no: The task number identifies the task for which the model is to be saved
    3) epoch_accuracy: Save the performance of the model on the task

    Function: Saves a reference for the classification head and the shared model at the
    appropriate locations

    """

    # create the variables

    # get the features of the classification head
    # in_features = model.tmodel.classifier[-1].in_features
    # out_features = model.tmodel.classifier[-1].out_features

    # 针对 SRNet 的情况
    classifier = model.tmodel.fc
    in_features = classifier.in_features
    out_features = classifier.out_features

    # separate out the classification head from the model
    ref = ClassificationHead(in_features, out_features)
    ref.fc.weight.data = classifier.weight.data
    ref.fc.bias.data = classifier.bias.data

    # save the reg_params
    reg_params = model.reg_params
    # print("reg_params = {}".format(reg_params))

    f = open(os.path.join(os.getcwd(), "models", "reg_params.pickle"), 'wb')
    pickle.dump(reg_params, f)
    f.close()

    # save tge model
    # del classifier

    # save the model at the specified location
    # torch.save(model.state_dict(), os.path.join(path_to_model, "shared_model.pth"))
    torch.save(model, os.path.join(PATH_TO_MODEL, 'model_in_task_{}'.format(task_no) + '.pth.tar'))
    # save the classification head at the task directory
    path_to_head = os.path.join(PATH_TO_MODEL, "Task_" + str(task_no))
    torch.save(ref.state_dict(), os.path.join(path_to_head, "head.pth"))
    del ref


def save_performance(epoch_accuracy, task_no):
    path_to_head = os.path.join(PATH_TO_MODEL, "Task_" + str(task_no))
    # save the performance of the model on the task to later determine the forgetting metric
    with open(os.path.join(path_to_head, "performance.txt"), 'w') as file:
        # input_to_txtfile = str(epoch_accuracy.item())
        input_to_txtfile = str(epoch_accuracy.item() if torch.is_tensor(epoch_accuracy) else epoch_accuracy)
        file.write(input_to_txtfile)
        file.close()

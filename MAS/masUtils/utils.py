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


def init_reg_params(model, use_gpu, lambda_list, freeze_layers=None):
    """
    Input:
    :param model: A reference to the model that is being trained
    :param use_gpu: Set the flag to True if the model is to be trained on the GPU
    :param lambda_list:
    :param freeze_layers: A list containing the layers for which omega is not calculated. Useful in the
        case of computational limitations where computing the importance parameters for the entire model
        is not feasible

    Output:
    1) model: A dictionary containing importance weights (omega), init_val (keep a reference
    to the initial values of the parameters) for all trainable parameters is calculated and the updated
    model with these reg_params is returned.


    Function: Initializes the reg_params for a model for the initial task (task = 1)

    """
    if freeze_layers is None:
        freeze_layers = []
    device = torch.device("cuda:0" if use_gpu else "cpu")

    reg_params = {}

    # model_layer_length = sum(1 for _ in model.tmodel.named_parameters())
    lambda_list_length = len(lambda_list)

    # if lambda_list is None:
    #     lambda_list = [1] * model_layer_length

    for index, (name, param) in enumerate(model.tmodel.named_parameters()):
        if name not in freeze_layers:
            print("Initializing omega values for layer", name)
            omega = torch.zeros(param.size())
            omega = omega.to(device)
            omega_list = [omega]
            init_val = param.data.clone()
            if index >= lambda_list_length:
                raise RuntimeError("index {} out of lambda_list_length {}".format(index, lambda_list_length))
            param_dict = {'omega': omega, 'init_val': init_val, 'lambda': lambda_list[index], 'omega_list': omega_list}

            # for first task, omega is initialized to zero

            # the key for this dictionary is the name of the layer
            reg_params[param] = param_dict

    model.reg_params = reg_params

    return model


def init_reg_params_across_tasks(model, use_gpu, freeze_layers=None):
    """
    Input:
    1) model: A reference to the model that is being trained
    2) use_gpu: Set the flag to True if the model is to be trained on the GPU
    3) freeze_layers: A list containing the layers for which omega is not calculated. Useful in the
        case of computational limitations where computing the importance parameters for the entire model
        is not feasible

    Output:
    1) model: A dictionary containing importance weights (omega), init_val (keep a reference
    to the initial values of the parameters) for all trainable parameters is calculated and the updated
    model with these reg_params is returned.


    Function: Initializes the reg_params for a model for other tasks in the sequence (task != 1)
    """

    # Get the reg_params for the model

    if freeze_layers is None:
        freeze_layers = []
    device = torch.device("cuda:0" if use_gpu else "cpu")

    reg_params = model.reg_params

    for name, param in model.tmodel.named_parameters():

        if name not in freeze_layers:

            if param in reg_params:
                param_dict = reg_params[param]
                print("Initializing the omega values for layer for the new task", name)

                # Store the previous values of omega
                prev_omega = param_dict['omega']

                # Initialize a new omega
                new_omega = torch.zeros(param.size())
                new_omega = new_omega.to(device)

                init_val = param.data.clone()
                init_val = init_val.to(device)

                param_dict['prev_omega'] = prev_omega
                param_dict['omega'] = prev_omega

                # store the initial values of the parameters
                param_dict['init_val'] = init_val

                omega_list = param_dict['omega_list']
                omega_list.append(new_omega)
                param_dict['omega_list'] = omega_list

                # the key for this dictionary is the name of the layer
                reg_params[param] = param_dict

    model.reg_params = reg_params

    return model


def consolidate_reg_params(model, use_gpu):
    """
    Input:
    1) model: A reference to the model that is being trained
    2) use_gpu: Set the flag to True if you wish to train the model on a GPU

    Output:
    1) reg_params: A dictionary containing importance weights (omega), init_val (keep a reference
    to the initial values of the parameters) for all trainable parameters


    Function: This function updates the value (adds the value) of omega across the tasks that the model is
    exposed to

    """
    # Get the reg_params for the model
    reg_params = model.reg_params

    for name, param in model.tmodel.named_parameters():
        if param in reg_params:
            param_dict = reg_params[param]
            print("Consolidating the omega values for layer", name)

            # Store the previous values of omega
            prev_omega = param_dict['prev_omega']
            new_omega = param_dict['omega']

            new_omega = torch.add(prev_omega, new_omega)
            del param_dict['prev_omega']

            param_dict['omega'] = new_omega

            # the key for this dictionary is the name of the layer
            reg_params[param] = param_dict

    model.reg_params = reg_params

    return model


def compute_omega_grads_norm(model, dataloader, optimizer, use_gpu):
    """
    Inputs:
    1) model: A reference to the model for which omega is to be calculated
    2) dataloader: A dataloader to feed the sample to the model
    3) optimizer: An instance of the "omega_update" class
    4) use_gpu: Flag is set to True if the model is to be trained on the GPU

    Outputs:
    1) model: An updated reference to the model is returned

    Function: Global version for computing the l2 norm of the function (neural network's) outputs. In
    addition to this, the function also accumulates the values of omega across the items of a task

    """
    # Alexnet object
    model.tmodel.eval()

    index = 0
    for sample in dataloader:

        # get the inputs and labels
        # inputs, labels = sample
        inputs, labels = sample['data'], sample['label']
        shape = list(inputs.size())
        inputs = inputs.reshape(shape[0] * shape[1], *shape[2:])
        labels = labels.reshape(-1)
        # shuffle
        idx = torch.randperm(shape[0])
        inputs = inputs[idx]
        labels = labels[idx]

        device = torch.device("cuda:0" if use_gpu else "cpu")
        if use_gpu:
            inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # get the function outputs
        outputs = model.tmodel(inputs)
        del inputs

        # compute the sqaured l2 norm of the function outputs
        l2_norm = torch.norm(outputs, 2, dim=1)
        del outputs

        squared_l2_norm = l2_norm ** 2
        del l2_norm

        sum_norm = torch.sum(squared_l2_norm)
        del squared_l2_norm

        # compute gradients for these parameters
        sum_norm.backward()

        # optimizer.step computes the omega values for the new batches of sample
        optimizer.step(model.reg_params, index, labels.size(0), use_gpu)
        del labels

        index = index + 1

    return model


# need a different function for grads vector
def compute_omega_grads_vector(model, dataloader, optimizer, use_gpu):
    """
    Inputs:
    1) model: A reference to the model for which omega is to be calculated
    2) dataloader: A dataloader to feed the data to the model
    3) optimizer: An instance of the "omega_update" class
    4) use_gpu: Flag is set to True if the model is to be trained on the GPU

    Outputs:
    1) model: An updated reference to the model is returned

    Function: This function backpropagates across the dimensions of the  function (neural network's)
    outputs. In addition to this, the function also accumulates the values of omega across the items
    of a task. Refer to section 4.1 of the paper for more details regarding this idea

    """

    # Alexnet object
    model.tmodel.train(False)
    model.tmodel.eval(True)

    index = 0

    for dataloader in dset_loaders:
        for data in dataloader:

            # get the inputs and labels
            inputs, labels = data

            if (use_gpu):
                device = torch.device("cuda:0")
                inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # get the function outputs
            outputs = model.tmodel(inputs)

            for unit_no in range(0, outputs.size(1)):
                ith_node = outputs[:, unit_no]
                targets = torch.sum(ith_node)

                # final node in the layer
                if (node_no == outputs.size(1) - 1):
                    targets.backward()
                else:
                    # This retains the computational graph for further computations
                    targets.backward(retain_graph=True)

                optimizer.step(model.reg_params, False, index, labels.size(0), use_gpu)

                # necessary to compute the correct gradients for each batch of data
                optimizer.zero_grad()

            optimizer.step(model.reg_params, True, index, labels.size(0), use_gpu)
            index = index + 1

    return model


# sanity check for the model to check if the omega values are getting updated
def sanity_model(model):
    for name, param in model.tmodel.named_parameters():

        print(name)

        if param in model.reg_params:
            param_dict = model.reg_params[param]
            omega = param_dict['omega']

            print("Max omega is", omega.max())
            print("Min omega is", omega.min())
            print("Mean value of omega is", omega.min())


# function to freeze selected layers
def create_freeze_layers(model, num_freeze_layers=2):
    # SRNet没有 classifier，features层。不实现此方法，直接返回空
    return [model, []]

    """
    Inputs
    1) model: A reference to the model
    2) num_freeze_layers: The number of convolutional layers that you want to freeze in the convolutional base of
        Alexnet model. Default value is 2

    Outputs
    1) model: An updated reference to the model with the requires_grad attribute of the
              parameters of the freeze_layers set to False
    2) freeze_layers: Creates a list of layers that will not be involved in the training process

    Function: This function creates the freeze_layers list which is then passed to the `compute_omega_grads_norm`
    function which then checks the list to see if the omegas need to be calculated for the parameters of these layers

    """

    # 原始代码的tmodel用的是alexNet，有自己的features和classifier，SRNet没有这些。
    # 而且为什么要把features层的requires_grad设置为false?可能是因为原始任务只需要alexNet进行迁移学习，所以不需要训练特征提取层
    # The require_grad attribute for the parameters of the classifier layer is set to True by default
    # for param in model.tmodel.classifier.parameters():
    #     param.requires_grad = True
    #
    # for param in model.tmodel.features.parameters():
    #     param.requires_grad = False

    # return an empty list if you want to train the entire model
    if num_freeze_layers == 0:
        return [model, []]

    temp_list = []
    freeze_layers = []

    # get the keys for the conv layers in the model
    # modules = model.tmodel.features._modules
    modules = model.tmodel._modules
    for key in modules:
        if type(modules[key]) == torch.nn.modules.conv.Conv2d:
            temp_list.append(key)

    # set the requires_grad attribute to True for the layers you want to be trainable
    """
    根据作者的Git记录，后来代码改成了如下这样，但是逻辑上似乎不对，不知道出于什么原因改的，暂且在此处以注释形式保留
    num_of_frozen_layers = len(temp_list) - num_freeze_layers
    
    for num in range(0, num_of_frozen_layers):
        # pick the layers from the end
        temp_key = temp_list[num]
    """
    for num in range(1, num_freeze_layers + 1):
        # pick the layers from the end
        temp_key = temp_list[-1 * num]

        for param in model.tmodel.features[int(temp_key)].parameters():
            param.requires_grad = True

        name_1 = 'features.' + temp_key + '.weight'
        name_2 = 'features.' + temp_key + '.bias'

        freeze_layers.append(name_1)
        freeze_layers.append(name_2)

    return [model, freeze_layers]

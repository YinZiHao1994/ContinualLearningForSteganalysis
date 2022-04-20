#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn.functional as F
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
            first_derivative = torch.zeros(param.size())
            first_derivative = first_derivative.to(device)
            first_derivative_list = [first_derivative]
            second_derivative = torch.zeros(param.size())
            init_val = param.data.clone()
            if index >= lambda_list_length:
                raise RuntimeError("index {} out of lambda_list_length {}".format(index, lambda_list_length))
            param_dict = {'omega': omega, 'omega_list': omega_list,
                          'first_derivative': first_derivative, 'first_derivative_list': first_derivative_list,
                          'second_derivative': second_derivative,
                          'init_val': init_val, 'lambda': lambda_list[index]}

            # for first task, omega is initialized to zero

            # the key for this dictionary is the name of the layer
            reg_params[param] = param_dict

    model.reg_params = reg_params

    return model


def init_reg_params_across_tasks(model, task_no, use_gpu, freeze_layers=None):
    """
    Input:
    :param model: A reference to the model that is being trained
    :param task_no:
    :param use_gpu: Set the flag to True if the model is to be trained on the GPU
    :param freeze_layers: A list containing the layers for which omega is not calculated. Useful in the
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
                first_derivative = param_dict['first_derivative']

                # Initialize a new omega
                new_omega = torch.zeros(param.size())
                new_omega = new_omega.to(device)

                init_val = param.data.clone()
                init_val = init_val.to(device)

                param_dict['prev_omega'] = prev_omega
                param_dict['omega'] = prev_omega

                # store the initial values of the parameters
                if task_no == 2:
                    param_dict['init_val'] = init_val

                omega_list = param_dict['omega_list']
                omega_list.append(new_omega)
                param_dict['omega_list'] = omega_list

                first_derivative_list = param_dict['first_derivative_list']
                new_first_derivative = torch.zeros(param.size())
                first_derivative_list.append(new_first_derivative)
                param_dict['first_derivative_list'] = first_derivative_list

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
    model.tmodel.eval()
    dataloader_len = len(dataloader)
    for index, sample in enumerate(dataloader):
        if index % 50 == 0:
            print("OmegaUpdate sample {}/{} in dataloader_train".format(index, dataloader_len))
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
        else:
            inputs, labels = Variable(inputs), Variable(labels)
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
        # sum_norm.backward(create_graph=True)
        # optimizer.step computes the omega values for the new batches of sample
        # optimizer.step(model.reg_params, index, labels.size(0), dataloader_len, use_gpu, 1)

        # 二阶导数的计算
        param_groups = optimizer.param_groups
        if len(param_groups) > 1:
            raise RuntimeError('param_groups length is {}'.format(len(param_groups)))
        params = param_groups[0]['params']
        filter_parms = []
        for param in params:
            if param.requires_grad is not None and param.requires_grad:
                filter_parms.append(param)
        # filter_parms = filter(lambda p: (p.requires_grad is not None and p.requires_grad), params)
        one_order_gradients = torch.autograd.grad(outputs=sum_norm, inputs=filter_parms, create_graph=True)
        deal_with_derivative(model, index, dataloader_len, labels.size(0), filter_parms, one_order_gradients, 1,
                             use_gpu)

        # torch.autograd.grad does not accumuate the gradients into the .grad attributes
        # It instead returns the gradients as Variable tuples.
        del sum_norm
        # now compute the 2-norm of the grad_params
        grad_norm = 0
        for grad in one_order_gradients:
            grad_norm += grad.pow(2).sum()
        grad_norm = grad_norm.sqrt()
        del one_order_gradients

        # take the gradients wrt grad_norm. backward() will accumulate
        # the gradients into the .grad attributes
        # grad_norm.backward()
        # optimizer.step(model.reg_params, index, labels.size(0), dataloader_len, use_gpu, 2)
        two_order_gradients = torch.autograd.grad(outputs=grad_norm, inputs=filter_parms)
        deal_with_derivative(model, index, dataloader_len, labels.size(0), filter_parms, two_order_gradients, 2,
                             use_gpu)

        # param_groups = optimizer.param_groups
        # if len(param_groups) > 1:
        #     raise RuntimeError('param_groups length is {}'.format(len(param_groups)))
        # params = param_groups[0]['params']
        #
        # one_order_gradients = torch.autograd.grad(outputs=sum_norm, inputs=params,
        #                                           grad_outputs=torch.ones(sum_norm.size()),
        #                                           retain_graph=True, create_graph=True)[0]
        #
        # deal_with_derivative(model, index, dataloader_len, labels.size(0), params, one_order_gradients, 1, use_gpu)
        #
        # two_order_gradients = torch.autograd.grad(outputs=one_order_gradients, inputs=params,
        #                                           grad_outputs=torch.ones(one_order_gradients.size()),
        #                                           create_graph=False)[0]
        # deal_with_derivative(model, index, dataloader_len, labels.size(0), params, two_order_gradients, 2, use_gpu)

        # one_order_gradients.backward(create_graph=False)
        # optimizer.step(model.reg_params, index, labels.size(0), dataloader_len, use_gpu, 2)
        del labels

    return model


def deal_with_derivative(model, batch_index, dataloader_len, batch_size, params, gradients, derivative_order, use_gpu):
    reg_params = model.reg_params
    params_length = len(params)
    gradients_length = len(gradients)
    if not params_length == gradients_length:
        raise RuntimeError("params_length {} not equal gradients_length {}".format(params_length, gradients_length))
    for param_index, param in enumerate(params):
        if param in reg_params:
            grad = gradients[param_index]
            if grad is None:
                print("param index {} grad is None".format(param_index))
                continue
            # The absolute value of the grad_data that is to be added to first_derivative
            grad_data_copy = grad.clone()
            # grad_data_copy = grad_data_copy.abs()

            param_dict = reg_params[param]
            if derivative_order == 1:
                first_derivative = param_dict['first_derivative']
                first_derivative = first_derivative.to(torch.device("cuda:0" if use_gpu else "cpu"))

                current_size = (batch_index + 1) * batch_size
                prev_size = batch_index * batch_size
                step_size = 1 / float(current_size)

                # Incremental update for the first_derivative
                # sum up the magnitude of the gradient
                new_first_derivative = ((first_derivative.mul(prev_size)).add(grad_data_copy)).div(current_size)
                param_dict['first_derivative'] = new_first_derivative
                if batch_index == dataloader_len - 1:
                    first_derivative_list = param_dict['first_derivative_list']
                    first_derivative_list[-1] = new_first_derivative
                    param_dict['first_derivative_list'] = first_derivative_list

                # if batch_index % 10 == 0:
                # print("in index {} ,param {}'s old first_derivative is {}\nnew first_derivative is {}"
                #       .format(batch_index, p, first_derivative, new_first_derivative))

                reg_params[param] = param_dict
                # 优化器的梯度是自动累加的，求完一阶导数后要清空tensor的grad，否则二阶导数的值会在一阶导数的基础上相加
                # p.grad.data.zero_()
            elif derivative_order == 2:
                second_derivative = param_dict['second_derivative']
                second_derivative = second_derivative.to(torch.device("cuda:0" if use_gpu else "cpu"))
                current_size = (batch_index + 1) * batch_size
                prev_size = batch_index * batch_size
                step_size = 1 / float(current_size)
                new_second_derivative = ((second_derivative.mul(prev_size)).add(grad_data_copy)).div(
                    current_size)
                param_dict['second_derivative'] = new_second_derivative
                if batch_index == dataloader_len - 1:
                    # calculate curvature
                    new_second_derivative = new_second_derivative.abs()
                    first_derivative = param_dict['first_derivative']
                    bottom = (1 + first_derivative ** 2) ** (3.0 / 2)
                    curvature = new_second_derivative / bottom
                    omega_list = param_dict['omega_list']
                    # omega = first_derivative.abs() * torch.log(curvature + 1)
                    omega = first_derivative.abs() * curvature
                    # print("first_derivative = {} ,new_second_derivative = {} ,curvature = {} ,omega = {}"
                    #       .format(first_derivative, new_second_derivative, curvature, omega))
                    print("max first_derivative = {} ,min first_derivative = {}"
                          .format(first_derivative.max(), first_derivative.min()))
                    print("max new_second_derivative = {} ,min new_second_derivative = {}"
                          .format(new_second_derivative.max(), new_second_derivative.min()))
                    print("max curvature = {} ,min curvature = {}"
                          .format(curvature.max(), curvature.min()))
                    print("max omega = {} ,min omega = {}"
                          .format(omega.max(), omega.min()))
                    omega_list[-1] = omega
                    param_dict['omega_list'] = omega_list
                reg_params[param] = param_dict
            else:
                raise RuntimeError("derivative_order ={} undefined".format(derivative_order))
        else:
            print("param index {} not in reg_params".format(param_index))


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

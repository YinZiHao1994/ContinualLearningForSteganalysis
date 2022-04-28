#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

import os
import shutil


# The idea is to have classification layers for different tasks


# class specific features are only limited to the last linear layer of the model
class ClassificationHead(nn.Module):
    """

    Each task has a seperate classification head which houses the features that
    are specific to that particular task. These features are unshared across tasks
    as described in section 5.1 of the paper

    """

    def __init__(self, in_features, out_features):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return x


used_omega_weight = 0.5
max_omega_weight = 0.5
prior_lambda = 3
later_lambda = 7


class SharedModel(nn.Module):

    def __init__(self, model):
        super(SharedModel, self).__init__()
        self.tmodel = model
        self.reg_params = {}
        # self.used_omega_weight = torch.tensor([0.3], requires_grad=True, dtype=torch.float64)
        # self.max_omega_weight = torch.tensor([0.3], requires_grad=True, dtype=torch.float64)
        self.used_omega_weight = torch.tensor(used_omega_weight, requires_grad=False)
        self.max_omega_weight = torch.tensor(max_omega_weight, requires_grad=False)
        # self.weight_params = {'used_omega_weight': used_omega_weight, 'max_omega_weight': max_omega_weight}
        self.lambda_list = init_lambda_list(self)

    def forward(self, x):
        return self.tmodel(x)


def init_lambda_list(model):
    # 每一层单独设定lambda
    model_layer_length = sum(1 for _ in model.tmodel.named_parameters())
    prior_length = 0
    lambda_list = []
    for index, (name, param) in enumerate(model.tmodel.named_parameters()):
        if name == 'bn72.bias':
            prior_length = index
            lambda_list = [prior_lambda] * (prior_length + 1)
    lambda_list.extend([later_lambda] * (model_layer_length - prior_length - 1))
    print(lambda_list)
    return lambda_list


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

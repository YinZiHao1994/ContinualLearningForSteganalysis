#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

import os
import shutil
from MAS.masUtils import utils, model_utils


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
        self.lambda_list = utils.create_lambda_list(self)

    def forward(self, x):
        return self.tmodel(x)


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

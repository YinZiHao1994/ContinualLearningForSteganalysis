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


class SharedModel(nn.Module):

    def __init__(self, model):
        super(SharedModel, self).__init__()
        self.tmodel = model
        self.reg_params = {}
        # self.used_omega_weight = torch.tensor([0.3], requires_grad=True, dtype=torch.float64)
        # self.max_omega_weight = torch.tensor([0.3], requires_grad=True, dtype=torch.float64)
        self.weight_params = {'used_omega_weight': torch.tensor(0.3, requires_grad=True, dtype=torch.float),
                              'max_omega_weight': torch.tensor(0.7, requires_grad=True, dtype=torch.float)}

    def forward(self, x):
        return self.tmodel(x)

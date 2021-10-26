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


class LocalSgd(optim.SGD):
    def __init__(self, params, reg_lambda, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(LocalSgd, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.reg_lambda = reg_lambda

    def __setstate__(self, state):
        super(LocalSgd, self).__setstate__(state)

    def step(self, reg_params, closure=None):

        loss = None
        use_gpu = torch.cuda.is_available()

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:

                if p.grad is None:
                    continue

                d_p = p.grad.data

                if p in reg_params:
                    param_dict = reg_params[p]

                    omega = param_dict['omega']
                    init_val = param_dict['init_val']
                    reg_lambda = param_dict['lambda']
                    curr_param_value = p.data
                    if use_gpu:
                        curr_param_value = curr_param_value.cuda()
                        init_val = init_val.cuda()
                        omega = omega.cuda()

                    # get the difference
                    param_diff = curr_param_value - init_val

                    # get the gradient for the penalty term for change in the weights of the parameters
                    # local_grad = torch.mul(param_diff, 2 * self.reg_lambda * omega)
                    # 使用每一层独立的reg_lambda替换整体设置的reg_lambda
                    local_grad = torch.mul(param_diff, 2 * reg_lambda * omega)
                    # print("omega = {} ,local_grad = {}".format(omega, local_grad))
                    # print("omega.min() = {} ,omega.max() = {} ,omega.mean() = {}"
                    #       .format(omega.min(), omega.max(), omega.mean()))
                    # print(
                    #     "local_grad.min() = {} ,local_grad.max() = {} ,local_grad.mean() = {}"
                    #         .format(local_grad.min(), local_grad.max(), local_grad.mean()))
                    del param_diff
                    del omega
                    del init_val
                    del curr_param_value

                    d_p = d_p + local_grad
                    # print("dp = {}".format(d_p))
                    # print("d_p.min() = {},d_p.max() = {} ,d_p.mean() = {}".format(d_p.min(), d_p.max(), d_p.mean()))
                    del local_grad

                if weight_decay != 0:
                    # d_p.add_(weight_decay, p.data)
                    d_p.add_(other=p.data, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        # buf.mul_(momentum).add_(1 - dampening, d_p)
                        buf.mul_(momentum).add_(other=d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # p.data.add_(-group['lr'], d_p)
                p.data.add_(other=d_p, alpha=-group['lr'])

        return loss


class OmegaUpdate(optim.SGD):

    def __init__(self, params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(OmegaUpdate, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def __setstate__(self, state):
        super(OmegaUpdate, self).__setstate__(state)

    def step(self, reg_params, batch_index, batch_size, use_gpu, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                if p in reg_params:
                    grad_data = p.grad.data

                    # The absolute value of the grad_data that is to be added to omega
                    grad_data_copy = p.grad.data.clone()
                    grad_data_copy = grad_data_copy.abs()

                    param_dict = reg_params[p]

                    omega = param_dict['omega']
                    omega = omega.to(torch.device("cuda:0" if use_gpu else "cpu"))

                    current_size = (batch_index + 1) * batch_size
                    step_size = 1 / float(current_size)

                    # Incremental update for the omega
                    new_omega = omega + step_size * (grad_data_copy - batch_size * omega)
                    # if batch_index % 10 == 0:
                    #     print("in index {} ,param {}'s old omega is {}\nnew omega is {}"
                    #           .format(batch_index, p, omega, new_omega))

                    param_dict['omega'] = new_omega

                    reg_params[p] = param_dict

        return loss


class OmegaVectorUpdate(optim.SGD):

    def __init__(self, params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(OmegaVectorUpdate, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def __setstate__(self, state):
        super(OmegaVectorUpdate, self).__setstate__(state)

    def step(self, reg_params, finality, batch_index, batch_size, use_gpu, closure=None):
        loss = None

        device = torch.device("cuda:0" if use_gpu else "cpu")

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                if p in reg_params:

                    grad_data = p.grad.data

                    # The absolute value of the grad_data that is to be added to omega
                    grad_data_copy = p.grad.data.clone()
                    grad_data_copy = grad_data_copy.abs()

                    param_dict = reg_params[p]

                    if not finality:

                        if 'temp_grad' in reg_params.keys():
                            temp_grad = param_dict['temp_grad']

                        else:
                            temp_grad = torch.FloatTensor(p.data.size()).zero_()
                            temp_grad = temp_grad.to(device)

                        temp_grad = temp_grad + grad_data_copy
                        param_dict['temp_grad'] = temp_grad

                        # del temp_data
                    else:

                        # temp_grad variable
                        temp_grad = param_dict['temp_grad']
                        temp_grad = temp_grad + grad_data_copy

                        # omega variable
                        omega = param_dict['omega']
                        omega.to(device)

                        current_size = (batch_index + 1) * batch_size
                        step_size = 1 / float(current_size)

                        # Incremental update for the omega
                        omega = omega + step_size * (temp_grad - batch_size * (omega))

                        param_dict['omega'] = omega

                        reg_params[p] = param_dict

                        del omega
                        del param_dict

                    del grad_data
                    del grad_data_copy

        return loss

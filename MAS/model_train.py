#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable

import copy
import os
import shutil

import sys
import time

# sys.path.append('masUtils')
# print(sys.path)
from MAS.masUtils.model_utils import *
from MAS.masUtils.utils import *
from MAS.model_class import *

from MAS.optimizer_lib import *
import dataAnalyze


def train_model(model, task_no, num_classes, model_criterion, dataloader_train, dataloader_valid, num_epochs,
                use_gpu=False, lr=0.001, reg_lambda=0.01, use_awl=False):
    """
    Outputs:
    1) model: Return a trained model

    Function: Trains the model on a specific task identified by a task number and saves this model
    :param model: A reference to the model that is being exposed to the sample for the task
    :param optimizer: A local_sgd optimizer object that implements the idea of MaS
    :param model_criterion: The loss function used to train the model
    :param dataloader_train: A dataloader to feed the training sample to the model
    :param dataloader_valid: A dataloader to feed the test sample to the model
    :param dset_size_train: Size of the dataset that belongs to a specific task
    :param dset_size_test: Size of the test dataset that belongs to a specific task
    :param num_epochs: Number of epochs that you wish to train the model for
    :param use_gpu: Set the flag to `True` if you wish to train on a GPU. Default value: False
    :param lr: The initial learning rate set for training the model
    :param use_awl:

    """
    omega_epochs = num_epochs + 1

    store_path = os.path.join(os.getcwd(), "models", "Task_" + str(task_no))
    model_path = os.path.join(os.getcwd(), "models")

    device = torch.device("cuda:0" if use_gpu else "cpu")

    # create a models directory if the directory does not exist
    if task_no == 1 and not os.path.isdir(model_path):
        os.mkdir(model_path)

    # the flag indicates that the the directory exists
    checkpoint_file, flag = check_checkpoints(store_path)

    start_epoch = 0

    if not flag:
        # create a task directory where the checkpoint files and the classification head will be stored
        create_task_dir(task_no, num_classes, store_path)
    else:
        ####################### Get the checkpoint if it exists ###############################

        # check for a checkpoint file
        if checkpoint_file == "":
            start_epoch = 0

        else:
            print("Loading checkpoint '{}' ".format(checkpoint_file))
            raise RuntimeError('Loading checkpoint')
            # checkpoint = torch.load(checkpoint_file)
            # start_epoch = checkpoint['epoch']
            #
            # print("Loading the model")
            # model = model_init(task_no, num_classes, use_gpu, True)
            # model = model.load_state_dict(checkpoint['state_dict'])
            #
            # print("Loading the optimizer")
            # optimizer = LocalSgd(model.reg_params, reg_lambda, model.weight_params)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            #
            # print("Done")

    ######################################################################################

    model.tmodel.train(True)
    model.tmodel.to(device)

    # commencing the training loop
    epoch_accuracy = 0
    loss_history = {x: [] for x in ['train', 'val']}
    acc_history = {x: [] for x in ['train', 'val']}
    counter = {x: [] for x in ['train', 'val']}
    iteration_number = {x: 0 for x in ['train', 'val']}
    epoch_number = {x: 0 for x in ['train', 'val']}

    since_time = time.time()

    automatic_weighted_loss = AutomaticWeightedLoss(2)
    momentum = 0.9
    weight_decay = 0.0005

    filter_parms = filter(lambda p: (p.requires_grad is not None and p.requires_grad) or p.requires_grad is None,
                          model.tmodel.parameters())
    params_wd, params_rest = [], []
    for name, param_item in model.named_parameters():
        if param_item.requires_grad:
            (params_wd if param_item.dim() != 1 else params_rest).append(param_item)
        else:
            print("{} not requires_grad".format(name))

    if use_awl:
        param_groups = [{'params': params_wd, 'weight_decay': weight_decay},
                        {'params': params_rest}, {'params': automatic_weighted_loss.parameters()}]
    else:
        param_groups = [{'params': params_wd, 'weight_decay': weight_decay},
                        {'params': params_rest}]

    optimizer = optim.SGD(param_groups, lr=lr, momentum=momentum, weight_decay=0.0005)

    step_size = 15
    scheduler_gama = 0.40
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=scheduler_gama)

    for epoch in range(start_epoch, omega_epochs):

        reg_params = model.reg_params
        if epoch == omega_epochs - 1:
            # run the omega accumulation at convergence of the loss function
            # no training of the model takes place in this epoch
            optimizer_ft = OmegaUpdate(model.reg_params)
            print("Updating the omega values for this task")
            model = compute_omega_grads_norm(model, dataloader_train, optimizer_ft, use_gpu)

            ############ 打印查看最大和最小的omega #############
            param_omega_list = []
            max_omega = None
            min_omega = None
            torch_max = 0
            torch_min = 100000
            for reg_param in reg_params:
                reg_param = reg_params[reg_param]
                param_omega = reg_param['omega']
                # param_omega_list.append(param_omega)
                torch_max = max(torch_max, torch.max(param_omega))
                torch_min = min(torch_min, torch.min(param_omega))
                # if max_omega is None:
                #     max_omega = torch.zeros_like(param_omega)
                # if min_omega is None:
                #     min_omega = torch.zeros_like(param_omega)
                # max_omega = torch.maximum(max_omega, param_omega)
                # min_omega = torch.minimum(min_omega, param_omega)
            # print("max_omega = {}\nmin_omega = {}".format(max_omega, min_omega))
            print("compute_omega_grads_norm max_omega = {} min_omega = {}".format(torch_max, torch_min))
        else:
            phase = 'train'

            total = 0
            best_perform = 10e6
            # print("Epoch {}/{}".format(epoch, num_epochs))
            print("-" * 20)
            # print ("The training phase is ongoing")
            running_loss = 0
            running_corrects = 0.0
            # scales the optimizer every 20 epochs
            # scheduler = exp_lr_scheduler(optimizer, epoch, lr, 40)

            model.tmodel.train(True)

            for index, sample in enumerate(dataloader_train):
                if index % 50 == 0:
                    print("sample {}/{} in dataloader_train".format(index, len(dataloader_train)))

                datas, labels = sample['data'], sample['label']
                shape = list(datas.size())
                datas = datas.reshape(shape[0] * shape[1], *shape[2:])
                labels = labels.reshape(-1)
                # shuffle
                idx = torch.randperm(shape[0])
                data = datas[idx]
                label = labels[idx]

                if use_gpu:
                    data = data.to(device)
                    label = label.to(device)

                else:
                    data = Variable(data)
                    label = Variable(label)

                # if index % 50 == 0:
                # weight_params = model.weight_params
                # used_omega_weight = weight_params['used_omega_weight']
                # max_omega_weight = weight_params['max_omega_weight']
                # used_omega_weight = model.used_omega_weight
                # max_omega_weight = model.max_omega_weight
                # print("used_omega_weight = {} sigmoid（used_omega_weight） = {},"
                #       .format(used_omega_weight, torch.sigmoid(used_omega_weight)))
                # print("used_omega_weight.requires_grad = {} used_omega_weight.grad = {},".format(
                #     model.used_omega_weight.requires_grad, model.used_omega_weight.grad))

                model.tmodel.to(device)
                optimizer.zero_grad()

                output = model.tmodel(data)
                del data

                origin_loss = model_criterion(output, label)
                # loss = origin_loss + regulation
                if task_no == 1:
                    loss = origin_loss
                    if index % 100 == 0:
                        print("loss = {}".format(loss))
                else:
                    regulation = calculate_regulation(model, reg_params, use_gpu)
                    if use_awl:
                        loss = automatic_weighted_loss(origin_loss, regulation)
                    else:
                        loss = origin_loss + regulation
                    if index % 100 == 0:
                        print("origin_loss = {} regulation = {} loss = {}".format(origin_loss, regulation, loss))
                        # for i, lam in enumerate(model.lambda_list):
                        #     print("lambda in position {} is {}".format(i, lam))
                        if use_awl:
                            for b, batch in enumerate(automatic_weighted_loss.parameters()):
                                print("automatic_weighted_loss {} is {}".format(b, batch))

                loss.backward()
                # print (model.reg_params)

                # optimizer.step(model.reg_params)
                optimizer.step()

                data_num = label.size(0)
                running_loss += loss.item() * data_num
                # running_corrects += torch.sum(preds == labels.data)
                prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced

                corrects = np.sum(prediction[1].cpu().numpy() == label.cpu().numpy())
                running_corrects += corrects
                del labels
                total += data_num

                # if index % 10 == 0:
                #     iteration_number[phase] += 10
                #     counter[phase].append(iteration_number[phase])
                #     loss_history[phase].append(loss.item())
                #     acc_history[phase].append(corrects / data_num)

            scheduler.step()

            dataset_size = len(dataloader_train.dataset)
            epoch_loss = running_loss / total
            epoch_accuracy = running_corrects / total
            epoch_number[phase] += 1
            counter[phase].append(epoch_number[phase])
            loss_history[phase].append(epoch_loss)
            acc_history[phase].append(epoch_accuracy)
            diagram_save_path = os.path.join(os.getcwd(), "diagram", "Task_" + str(task_no))
            dataAnalyze.save_loss_plot(diagram_save_path, counter[phase], loss_history[phase],
                                       "loss_" + phase + "_" + str(epoch))
            dataAnalyze.save_accurate_plot(diagram_save_path, counter[phase], acc_history[phase],
                                           "acc_" + phase + "_" + str(epoch))

            print('train epoch: {}/{}\n'
                  'Loss: {:.4f}\n'
                  'Acc: {:.4f}\n'
                  'lr:{}'
                  .format(epoch + 1, num_epochs, epoch_loss, epoch_accuracy,
                          optimizer.state_dict()['param_groups'][0]['lr']))

            # avoid saving a file twice
            if epoch != 0 and epoch != num_epochs - 1 and (epoch + 1) % 10 == 0:
                epoch_file_name = os.path.join(store_path, str(epoch + 1) + '.pth.tar')
                torch.save({
                    'epoch': epoch,
                    'epoch_loss': epoch_loss,
                    'epoch_accuracy': epoch_accuracy,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),

                }, epoch_file_name)

            if (epoch != 0 and (epoch % (num_epochs / 5)) == 0) or epoch == (num_epochs - 1):
                # evaluate performance specified number
                phase = 'val'
                total = 0
                running_loss = 0
                running_corrects = 0.0

                model.tmodel.eval()
                with torch.no_grad():
                    for index, sample in enumerate(dataloader_valid):
                        if index % 100 == 0:
                            print("sample {}/{} in dataloader_test".format(index, len(dataloader_valid)))
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

                        # optimizer.zero_grad()

                        output = model.tmodel(data)
                        del data
                        loss = model_criterion(output, label)
                        # running_corrects += torch.sum(preds == labels.data)
                        prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced

                        corrects = np.sum(prediction[1].cpu().numpy() == label.cpu().numpy())
                        running_corrects += corrects
                        del labels
                        data_num = label.size(0)
                        total += data_num
                        if index % 10 == 0:
                            iteration_number[phase] += 10
                            counter[phase].append(iteration_number[phase])
                            loss_history[phase].append(loss.item())
                            acc_history[phase].append(corrects / data_num)
                diagram_save_path = os.path.join(os.getcwd(), "diagram", "Task_" + str(task_no))
                dataAnalyze.save_loss_plot(diagram_save_path, counter[phase], loss_history[phase],
                                           "loss_" + phase + "_" + str(epoch))
                dataAnalyze.save_accurate_plot(diagram_save_path, counter[phase], acc_history[phase],
                                               "acc_" + phase + "_" + str(epoch))
                dataset_size = len(dataloader_valid.dataset)
                epoch_accuracy = running_corrects / total
                print("valuate in epoch {} accuracy is {}".format(epoch, epoch_accuracy))

    time_elapsed = time.time() - since_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # save the model and the performance
    save_model(model, task_no)


def calculate_regulation(model, reg_params, use_gpu):
    # weight_params = model.weight_params
    # used_omega_weight = weight_params['used_omega_weight']
    # max_omega_weight = weight_params['max_omega_weight']
    used_omega_weight = model.used_omega_weight
    max_omega_weight = model.max_omega_weight
    parameters = model.tmodel.parameters()
    regulation = 0
    for index, p in enumerate(parameters):
        if p in reg_params:
            param_dict = reg_params[p]

            omega = param_dict['omega']
            omega_list = param_dict['omega_list']
            used_omega = 0
            omega_list_length = len(omega_list)
            max_omega = None

            for i, ome in enumerate(omega_list):
                #     used_omega = ome * (omega_list_length - i) + used_omega
                if max_omega is None:
                    max_omega = torch.zeros_like(ome)
                max_omega = torch.max(max_omega, ome)
                # 由于网络在训练中已经考虑了omega的影响，此处叠加omega，越早计算得到的omega占的权重应该越小
                used_omega = ome * (i + 1) / omega_list_length + used_omega
                # used_omega = ome + used_omega
                # if self.flag < 1:
                #     print("in LocalSgd ,ome_{} = {}".format(i, ome[:1, :, :]))
                #     print("in LocalSgd ,max_omega_{} = {}".format(i, max_omega[:1, :, :]))
            used_omega = used_omega / omega_list_length
            # used_omega_weight_sigmoid = torch.sigmoid(model.used_omega_weight)
            used_omega = used_omega * used_omega_weight + max_omega * max_omega_weight
            init_val = param_dict['init_val']
            reg_lambda = param_dict['lambda']
            # curr_param_value_copy = p.data.clone()
            curr_param_value_copy = p
            if use_gpu:
                curr_param_value_copy = curr_param_value_copy.cuda()
                init_val = init_val.cuda()
                used_omega = used_omega.cuda()
            # get the difference
            param_diff = curr_param_value_copy - init_val
            mul = torch.mul(param_diff ** 2, used_omega)
            regulation += reg_lambda * mul.sum()
        else:
            print("param in index {} not in reg_params".format(index))
    return regulation

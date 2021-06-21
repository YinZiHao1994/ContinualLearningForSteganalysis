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

# sys.path.append('./utils')
from utils.model_utils import *
from utils.mas_utils import *

from optimizer_lib import *


def train_model(model, task_no, num_classes, optimizer, model_criterion, dataloader_train, dataloader_test, num_epochs,
                use_gpu=False, lr=0.001, reg_lambda=0.01):
    """
    Inputs:
    1) model: A reference to the model that is being exposed to the sample for the task
    2) optimizer: A local_sgd optimizer object that implements the idea of MaS
    3) model_criterion: The loss function used to train the model
    4) dataloader_train: A dataloader to feed the training sample to the model
    5) dataloader_test: A dataloader to feed the test sample to the model
    6) dset_size_train: Size of the dataset that belongs to a specific task
    7) dset_size_test: Size of the test dataset that belongs to a specific task
    8) num_epochs: Number of epochs that you wish to train the model for
    9) use_gpu: Set the flag to `True` if you wish to train on a GPU. Default value: False
    10) lr: The initial learning rate set for training the model

    Outputs:
    1) model: Return a trained model

    Function: Trains the model on a specific task identified by a task number and saves this model

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
            checkpoint = torch.load(checkpoint_file)
            start_epoch = checkpoint['epoch']

            print("Loading the model")
            model = model_init(num_classes, use_gpu)
            model = model.load_state_dict(checkpoint['state_dict'])

            print("Loading the optimizer")
            optimizer = LocalSgd(model.reg_params, reg_lambda)
            optimizer.load_state_dict(checkpoint['optimizer'])

            print("Done")

    ######################################################################################

    model.tmodel.train(True)
    model.tmodel.to(device)

    # commencing the training loop
    epoch_accuracy = 0
    for epoch in range(start_epoch, omega_epochs):

        # run the omega accumulation at convergence of the loss function
        if epoch == omega_epochs - 1:
            # no training of the model takes place in this epoch
            optimizer_ft = OmegaUpdate(model.reg_params)
            print("Updating the omega values for this task")
            model = compute_omega_grads_norm(model, dataloader_train, optimizer_ft, use_gpu)

            running_loss = 0
            running_corrects = 0.0

            model.tmodel.eval()

            for index, sample in enumerate(dataloader_test):
                if index % 50 == 0:
                    print("sample {}/{} in dataloader_test".format(index, len(dataloader_test)))
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

                # running_corrects += torch.sum(preds == labels.data)
                prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced

                running_corrects += np.sum(prediction[1].cpu().numpy() == label.cpu().numpy())
                del labels

            dataset_size = len(dataloader_test.dataset)
            epoch_accuracy = running_corrects / dataset_size
            print("valuate epoch_accuracy is {}".format(epoch_accuracy))
        else:

            since = time.time()
            best_perform = 10e6

            # print("Epoch {}/{}".format(epoch + 1, num_epochs))
            print("-" * 20)
            # print ("The training phase is ongoing")

            running_loss = 0
            running_corrects = 0.0

            # scales the optimizer every 20 epochs
            optimizer = exp_lr_scheduler(optimizer, epoch, lr, 40)

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

                model.tmodel.to(device)
                optimizer.zero_grad()

                output = model.tmodel(data)
                del data

                loss = model_criterion(output, labels)

                loss.backward()
                # print (model.reg_params)

                optimizer.step(model.reg_params)

                running_loss += loss.item()
                del loss

                # running_corrects += torch.sum(preds == labels.data)
                prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced

                running_corrects += np.sum(prediction[1].cpu().numpy() == label.cpu().numpy())
                del labels

            dataset_size = len(dataloader_train.dataset)
            epoch_loss = running_loss / dataset_size
            epoch_accuracy = running_corrects / dataset_size

            print('train epoch: {}/{}\n'
                  'Loss: {:.4f}\n'
                  'Acc: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss, epoch_accuracy))

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

    # save the model and the performance
    save_model(model, task_no, epoch_accuracy)

#!/usr/bin/env python

import numpy as np, os, sys, joblib
from scipy.io import loadmat
import pandas as pd
import os, time

import torch
from torch import nn, optim

from config import config
from scipy import signal
import utils
import warnings
from ResNet import ResNet34
import copy

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(666)
torch.cuda.manual_seed(666)


# Load challenge data.
def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file, 'r') as f:
        header_data = f.readlines()

    return data, header_data


# Find unique classes.
def get_classes(input_directory, filenames):
    classes = set()
    for filename in filenames:
        input_file = os.path.join(input_directory, filename)
        with open(input_file, 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)


def train(x_train, x_train_external, y_train):
    # model

    num_class = np.shape(y_train)[1]

    model = ResNet34(num_classes=num_class)
    model = model.to(device)

    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    wc = y_train.sum(axis=0)
    wc = 1. / (np.log(wc+1) + 1)

    w = torch.tensor(wc, dtype=torch.float).to(device)
    criterion1 = utils.WeightedMultilabel(w)

    lr = config.lr
    start_epoch = 1
    stage = 1
    best_auc = -1

    # =========>开始训练<=========
    for epoch in range(start_epoch, config.max_epoch + 1):
        train_loss, train_auc = train_epoch(model, optimizer, criterion1, x_train, x_train_external, y_train)

        if epoch in config.stage_epoch:
            stage += 1
            lr /= config.lr_decay

            utils.adjust_learning_rate(optimizer, lr)
    return model


def train_epoch(model, optimizer, criterion, x_train, x_train_external, y_train):
    model.train()
    auc_meter, loss_meter, it_count = 0, 0, 0
    batch_size = config.batch_size

    for i in range(0, len(x_train) - batch_size + 1, batch_size):
        inputs1 = torch.tensor(x_train[i:i + batch_size], dtype=torch.float, device=device)
        inputs2 = torch.tensor(x_train_external[i:i + batch_size], dtype=torch.float, device=device)
        target = torch.tensor(y_train[i:i + batch_size], dtype=torch.float, device=device)
        output = model.forward(inputs1, inputs2)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        auc_meter = auc_meter + utils.calc_auc(target, torch.sigmoid(output))

    return loss_meter / it_count, auc_meter / it_count


def train_12ECG_classifier(input_directory, output_directory):
    input_files = []
    header_files = []

    train_directory = input_directory
    for f in os.listdir(train_directory):
        if os.path.isfile(os.path.join(train_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
            g = f.replace('.mat', '.hea')
            input_files.append(f)
            header_files.append(g)

    # the 27 scored classes
    classes_weight = ['270492004', '164889003', '164890007', '426627000', '713427006', '713426002', '445118002', '39732003',
                      '164909002', '251146004', '10370003', '284470004', '164947007', '111975006',
                      '47665007', '59118001', '427393009', '426177001', '426783006', '427084000', '63593006',
                      '17338001', '59931005', '164917005', '164934002', '427172004', '698252002']

    classes_name = sorted(classes_weight)
    num_files = len(input_files)
    num_class = len(classes_name)

    set_length = 10000
    data_num = np.zeros((num_files, 12, set_length))
    classes_num = np.zeros((num_files, num_class))
    data_external = np.zeros((num_files, 2))

    for cnt, f in enumerate(input_files):
        classes = set()
        tmp_input_file = os.path.join(train_directory, f)
        data, header_data = load_challenge_data(tmp_input_file)

        for i, lines in enumerate(header_data):
            if i == 0:
                line0 = lines.split(' ')
                rs = line0[2]
                if rs != 500:
                    tmp_data = []
                    nums = int(int(line0[3]) / int(line0[2]) * 500)
                    for i in range(data.shape[0]):
                        tmp_data.append(signal.resample(data[i], nums))
                    data = copy.deepcopy(np.array(tmp_data))
            if lines.startswith('#Age'):
                tmp_age = lines.split(': ')[1].strip()
                age = int(tmp_age if tmp_age != 'NaN' else 57)
                age = age / 100
            elif lines.startswith('#Sex'):
                tmp_sex = lines.split(': ')[1]
                if tmp_sex.strip() == 'Female':
                    sex = 1
                else:
                    sex = 0

            elif lines.startswith('#Dx'):
                tmp = lines.split(': ')[1].split(',')
                for c in tmp:
                    classes.add(c.strip())

                for j in classes:
                    if j in classes_name:
                        class_index = classes_name.index(j)
                        classes_num[cnt, class_index] = 1

        data_external[cnt, 0] = age
        data_external[cnt, 1] = sex

        data_lens = data.shape[1]
        if data_lens < set_length:
            data = np.pad(data, ((0, 0), (set_length - data_lens, 0)), mode='constant', constant_values=0)
        elif data_lens > set_length:
            data = data[:, :set_length]

        data = data_transform(data, train=True)
        data_num[cnt] = data
    classes_num = pd.DataFrame(classes_num, columns=classes_name, dtype='int')
    classes_num['713427006'] = classes_num['713427006'] | classes_num['59118001']
    classes_num['59118001'] = classes_num['713427006']
    classes_num['284470004'] = classes_num['284470004'] | classes_num['63593006']
    classes_num['63593006'] = classes_num['284470004']
    classes_num['427172004'] = classes_num['427172004'] | classes_num['17338001']
    classes_num['17338001'] = classes_num['427172004']
    classes_num = np.array(classes_num)

    model = train(data_num, data_external, classes_num)

    # save the model
    output_directory = os.path.join(output_directory, 'resnet_0824.pkl')
    torch.save(model, output_directory)


def scaling(sig, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma,
                                     size=(1, sig.shape[1]))
    myNoise = np.matmul(np.ones((sig.shape[0], 1)), scalingFactor)
    return sig * myNoise


def shift(sig, interval=50):
    for col in range(sig.shape[0]):
        offset = np.random.choice(range(-interval, interval))
        sig[col, :] += offset
    return sig


def data_transform(sig, train=False):
    if train:
        if np.random.randn() > 0.5:
            sig = scaling(sig)

        if np.random.randn() > 0.5:
            sig = shift(sig)

    return sig

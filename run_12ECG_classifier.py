#!/usr/bin/env python

import numpy as np, os, sys
import joblib
import torch
from scipy import signal
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_features(data, header_data):
    set_length = 10000
    data_external = np.zeros((1, 2))

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

    data_lens = data.shape[1]
    if data_lens < set_length:
        data = np.pad(data, ((0, 0), (set_length - data_lens, 0)), mode='constant', constant_values=0)
    elif data_lens > set_length:
        data = data[:, :set_length]

    data_num = data.reshape(1, 12, -1)

    data_external[:, 0] = age
    data_external[:, 1] = sex
    return data_num, data_external


def load_12ECG_model(input_directory):
    # load the model from disk
    f_out = 'resnet.pkl'
    filename = os.path.join(input_directory, f_out)
    loaded_model = torch.load(filename, map_location=device)
    return loaded_model


def run_12ECG_classifier(data, header_data, model):
    classes = ['270492004', '164889003', '164890007', '426627000', '713427006', '713426002', '445118002', '39732003',
               '164909002', '251146004', '10370003', '284470004', '164947007', '111975006', '47665007', '59118001',
               '427393009', '426177001', '426783006', '427084000', '63593006', '17338001', '59931005', '164917005',
               '164934002', '427172004', '698252002']

    classes = sorted(classes)
    num_classes = len(classes)

    # Use your classifier here to obtain a label and score for each class.
    feats_reshape, feats_external = get_features(data, header_data)

    feats_reshape = torch.tensor(feats_reshape, dtype=torch.float, device=device)
    feats_external = torch.tensor(feats_external, dtype=torch.float, device=device)

    pred = model.forward(feats_reshape, feats_external)
    pred = torch.sigmoid(pred)
    current_score = pred.squeeze().cpu().detach().numpy()

    current_label = np.where(current_score > 0.15, 1, 0)
    current_label = current_label.astype(int)

    num_positive_classes = np.sum(current_label)
    # 窦性心律标签处于有评分的标签排序后的第14位
    normal_index = classes.index('426783006')

    ##至少为一个标签，如果所有标签都没有，就将窦性心律设为1
    if num_positive_classes == 0:
        current_label[normal_index] = 1

    return current_label, current_score, classes

from collections import OrderedDict

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

""" Conf.py """
class myDotDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, val in dct.items():
            self[key] = val


""" Dropout """
def parse_dropout_strategy(dropout_strategy):
    dropout_list = dropout_strategy.split(',')
    turning_point_list = []
    value_list = []
    for dropout in dropout_list:
        turning_point, value = dropout.split('@')
        turning_point_list.append(int(turning_point))
        value_list.append(float(value))
    return turning_point_list, value_list


def cal_drop_p(epoch, dropout_strategy):
    turning_point_list, value_list = dropout_strategy
    idx = sum([int(epoch >= x) for x in turning_point_list])
    if idx == len(turning_point_list):
        return value_list[-1]
    else:
        return ((epoch - turning_point_list[idx - 1]) * 
                (value_list[idx] - value_list[idx - 1]) / (turning_point_list[idx] - turning_point_list[idx - 1])
                + value_list[idx - 1])


""" Plot """
def PlotSpectrogram(X, fs, frame_move):
    if not isinstance(X, np.ndarray):
        X = X.detach().cpu().numpy()
    X_plot = np.abs(X.squeeze())
    plt.figure()
    librosa.display.specshow(librosa.amplitude_to_db(X_plot, ref=np.max), y_axis='linear',
                             x_axis='time', sr=fs, hop_length=frame_move)
    plt.show()
    fig = plt.gcf()
    plt.close(fig=fig)
    return fig


""" Training context manager """
class training_manager(object):
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
    
    def reset_dict(self, keys, init_vals=None):
        dct = OrderedDict({})
        for i, key in enumerate(keys):
            dct[key] = init_vals[i] if init_vals is not None else 0.0
        return dct
    
    def accumulate_dict(self, dct, vals):
        for i, key in enumerate(dct.keys()):
            dct[key] += vals[i]

    def average_dict(self, dct, step):
        for i, key in enumerate(dct.keys()):
            dct[key] /= step

    def add2buffer(self, buffer, dct):
        for key, val in dct.items():
            buffer[key].append(val)

    def add2tb(self, writer, dct, sub_key, idx):
        for key, val in dct.items():
            writer.add_scalars(key, {sub_key: val}, idx)


""" CollateFn """
class Collate():
    def __init__(self, cfgs, batch):
        self.cfgs = cfgs
        if self.cfgs.info.stage == 1:
            data, label = list(zip(*batch))
            self.data = torch.stack(data, dim=0)
        elif self.cfgs.info.stage == 2:
            remark, data_target, data_noise, mixinfo, label = list(zip(*batch))
            self.remark = remark
            self.data_target = torch.stack(data_target, axis=0)
            self.data_noise = torch.stack(data_noise, axis=0)
            self.mixinfo = mixinfo
        self.label = torch.stack(label, dim=0)

    def pin_memory(self):
        if self.cfgs.info.stage == 1:
            self.data = self.data.pin_memory()
        elif self.cfgs.info.stage == 2:
            self.data_target = self.data_target.pin_memory()
            self.data_noise = self.data_noise.pin_memory()
        self.label = self.label.pin_memory()
        return self


class CollateFnClass():
    def __init__(self, cfgs):
        self.cfgs = cfgs

    def CollateFn(self, batch):
        return Collate(self.cfgs, batch)


class CollateNew():
    def __init__(self, cfgs, batch):
        self.cfgs = cfgs
        remark, data, mixinfo, label, nutt_per_spkr = list(zip(*batch))
        data = list(zip(*data))
        self.enroll_num = nutt_per_spkr[0][1]
        self.remark = remark
        self.mixinfo = mixinfo
        self.label = torch.stack(label, dim=0)
        self.data_target = torch.stack(data[0], axis=0).squeeze()
        self.data_enroll = torch.stack(data[1], axis=0)
        self.data_enroll = self.data_enroll.view(-1, *self.data_enroll.shape[2:])
        if self.cfgs.info.stage == 2:
            self.data_noise = torch.stack(data[2], axis=0).squeeze()

    def pin_memory(self):
        self.data_target = self.data_target.pin_memory()
        self.data_enroll = self.data_enroll.pin_memory()
        if self.cfgs.info.stage == 2:
            self.data_noise = self.data_noise.pin_memory()
        self.label = self.label.pin_memory()
        return self


class CollateFnClassNew():
    def __init__(self, cfgs):
        self.cfgs = cfgs

    def CollateFn(self, batch):
        return CollateNew(self.cfgs, batch)
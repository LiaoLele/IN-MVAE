import torch
# from config.hparam import hparam as hp
import torch.nn.functional as F
import numpy as np
import librosa as rosa
import torchaudio


# def my_spectrogram(sample, stft_len, stft_hop):  # sample: [nchannel, time]
#     sample = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40, dct_type=2, norm='ortho', log_mels=False, melkwargs=None)
#     sample = torch.stft(sample, stft_len, win_length=stft_len, hop_length=stft_hop,
#                         window=torch.hann_window(stft_len, device=sample.device))
#     sample = sample.pow(2).sum(-1)
#     sample_max, _ = sample.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)
#     return sample / sample_max

# def my_collate_fn(batch):
#     data, label = list(zip(*batch))
#     data = torch.cat(data, axis=0)
#     label = torch.cat(label, axis=0)
#     return data, label

class myDotDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, val in dct.items():
            self[key] = val


class SimpleCollate():
    def __init__(self, batch):
        data, label, uttinfo = list(zip(*batch))
        self.data = torch.stack(data, axis=0)
        self.label = torch.stack(label, axis=0)
        self.uttinfo = uttinfo

    def pin_memory(self):
        self.data = self.data.pin_memory()
        self.label = self.label.pin_memory()
        # print(self.data.is_pinned())
        # print(self.label.is_pinned)
        return self


def CollateFnWrapper(batch):
    return SimpleCollate(batch)


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

# n_fft = int(0.064 * 16000)
# hop = int(0.016 * 16000)
# melsetting = {}
# melsetting['n_fft'] = n_fft
# melsetting['win_length'] = n_fft
# melsetting['hop_length'] = hop
# melsetting['n_mels'] = 30
# transform = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=hp.model.feat_num, melkwargs=melsetting)


# def my_collate_fn(dataset, indices, transform_fn=None, **transform_fn_kwargs):
#     batch = dataset.get_batch_items(indices)
#     data, idx = zip(*batch)
#     data = torch.from_numpy(np.stack(data, axis=0))
#     idx = torch.from_numpy(np.array(idx))
#     with torch.no_grad():
#         data = transform(data.float())
#     return data, idx

import torch
import numpy as np


def my_spectrogram(sample, stft_len, stft_hop, multi_num=4):  # sample: [nchannel, time]
    sample = torch.stft(sample, stft_len, win_length=stft_len, hop_length=stft_hop,
                        window=torch.hann_window(stft_len, device=sample.device))
    sample = sample.pow(2).sum(-1)
    sample = sample[:, :, :sample.shape[-1] - sample.shape[-1] % multi_num]
    return sample


def spectrogram_normalize(sample):  #sample [channel, nfreq, time]
    sample_max, _ = sample.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)
    sample = sample / sample_max
    return sample


class SimpleCollate():
    def __init__(self, batch):
        data, label = list(zip(*batch))
        self.data = torch.stack(data, axis=0)
        self.label = torch.stack(label, axis=0)

    def pin_memory(self):
        self.data = self.data.pin_memory()
        self.label = self.label.pin_memory()
        return self


def CollateFnWrapper(batch):
    return SimpleCollate(batch)


def zero_pad(x, num_pad, hop_length=256):
    """ x: [n_channel, nsample] """
    if (x.shape[1] / hop_length + 1) % num_pad == 0:    # common equation is (x.shape[1]/frame_move + 1) % num_pad == 0, where num_pad is the required to conduct cnn
        return x
    rest = (num_pad * hop_length) - (x.shape[1] + hop_length) % (num_pad * hop_length)
    left = rest // 2
    right = rest - left
    return np.pad(x, ((0, 0), (left, right)), mode='constant')
import torch.nn.functional as F
import torch.nn as nn
import torch
from ..config.hparam import hparam as hp



class tdnn_block(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1):
        super(tdnn_block, self).__init__()
        self.tdnn = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups,
                              bias=True, padding_mode='zeros')
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(out_channel)

    def forward(self, x):  # x [N, C, T]
        return self.batchnorm(self.relu(self.tdnn(x)))


class linear_block(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(linear_block, self).__init__()
        self.fc = nn.Linear(in_channel, out_channel)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(out_channel)    # the batchnorm layer is very important

    def forward(self, x):  # x [N, C]
        return self.batchnorm(self.relu(self.fc(x)))


class StatsPooling(nn.Module):

    def forward(self, x):  # x [N, C, T]
        avg = x.mean(dim=-1)
        std = x.std(dim=-1)
        return torch.cat((avg, std), dim=-1)


class speaker_encoder_xvec(nn.Module):

    def __init__(self):
        super(speaker_encoder_xvec, self).__init__()
        self.training = self.training

        self.tdnn1 = tdnn_block(hp.model.feat_num, hp.model.tdnn_channels[0], 5,
                                stride=1, padding=0, dilation=1)
        self.tdnn2 = tdnn_block(hp.model.tdnn_channels[0], hp.model.tdnn_channels[1], 3,
                                stride=1, padding=0, dilation=2)
        self.tdnn3 = tdnn_block(hp.model.tdnn_channels[1], hp.model.tdnn_channels[2], 3,
                                stride=1, padding=0, dilation=3)
        self.tdnn4 = tdnn_block(hp.model.tdnn_channels[2], hp.model.tdnn_channels[3], 1,
                                stride=1, padding=0, dilation=1)
        self.tdnn5 = tdnn_block(hp.model.tdnn_channels[3], hp.model.tdnn_channels[4], 1,
                                stride=1, padding=0, dilation=1)
        self.statspool = StatsPooling()
        self.fc1 = linear_block(int(2 * hp.model.tdnn_channels[4]), hp.model.fc_channels[0])
        self.fc2 = linear_block(hp.model.fc_channels[0], hp.model.fc_channels[1])

    def extract_embd(self, x, num=1, use_slide=True):  # x [N, C, T]
        batchsize = x.shape[0]
        if use_slide:
            x = self.slide(x)
        h = self.tdnn1(x)
        h = self.tdnn2(h)
        h = self.tdnn3(h)
        h = self.tdnn4(h)
        h = self.tdnn5(h)
        h = self.statspool(h)
        out_1 = self.fc1.fc(h)
        out_1 = out_1.view(-1, batchsize, out_1.shape[-1])
        out_1 = torch.mean(out_1, dim=0)
        # out_2 = self.fc2.fc(F.relu(out_1))
        # out_2 = out_2.view(-1, batchsize, out_2.shape[-1])
        # out_2 = torch.mean(out_2, dim=0)
        # return out_1 if num == 1 else out_2
        return out_1

    def slide(self, x):  # x: [batchsize, nfreq, time]
        frame_num = hp.train.spencoder_frame_num
        frame_hop = int(hp.train.spencoder_frame_hop * frame_num)
        sample_num = int((x.shape[-1] - frame_num) // frame_hop + 1)
        x_seg = []
        for i in range(sample_num):
            x_seg.append(x[:, :, i * frame_hop: i * frame_hop + frame_num])
        x_seg = torch.stack(x_seg, dim=0)
        x_seg = x_seg.view(-1, x_seg.shape[-2], x_seg.shape[-1])
        assert x_seg.shape[2] == hp.train.spencoder_frame_num
        return x_seg
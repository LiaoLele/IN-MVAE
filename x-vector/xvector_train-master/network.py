import torch
import torch.nn as nn
import torch.nn.functional as F

# from config.hparam import hparam as hp
from conf import cfgs


class tdnn_block(nn.Module): #tdnn: time-delay network

    def __init__(self, in_channel, out_channel, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1):
        super(tdnn_block, self).__init__()
        self.tdnn = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups,
                              bias=True, padding_mode='zeros')
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(out_channel)
        # The mean and standard-deviation are calculated per-dimension over the mini-batches
        # output has the same shape as input

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


class net(nn.Module):

    def __init__(self, out_dim, drop_p=None):
        super(net, self).__init__()
        self._drop_p = drop_p
        self.training = self.training
        self.out_dim = out_dim

        self.tdnn1 = tdnn_block(cfgs.model.feat_num, cfgs.model.tdnn_channels[0], 5,
                                stride=1, padding=0, dilation=1)
        self.tdnn2 = tdnn_block(cfgs.model.tdnn_channels[0], cfgs.model.tdnn_channels[1], 3,
                                stride=1, padding=0, dilation=2)
        self.tdnn3 = tdnn_block(cfgs.model.tdnn_channels[1], cfgs.model.tdnn_channels[2], 3,
                                stride=1, padding=0, dilation=3)
        self.tdnn4 = tdnn_block(cfgs.model.tdnn_channels[2], cfgs.model.tdnn_channels[3], 1,
                                stride=1, padding=0, dilation=1)
        self.tdnn5 = tdnn_block(cfgs.model.tdnn_channels[3], cfgs.model.tdnn_channels[4], 1,
                                stride=1, padding=0, dilation=1)
        self.statspool = StatsPooling()
        self.fc1 = linear_block(int(2 * cfgs.model.tdnn_channels[4]), cfgs.model.fc_channels[0])
        self.fc2 = linear_block(cfgs.model.fc_channels[0], cfgs.model.fc_channels[1])
        self.affine = nn.Linear(cfgs.model.fc_channels[1], self.out_dim)

    @property
    def drop_p(self):
        return self._drop_p

    @drop_p.setter
    def drop_p(self, p):
        assert p >= 0 and p < 1, "dropout probability is not valid"
        self._drop_p = p

    def forward(self, x):  # x [N, C, T]
        h = self.tdnn1(x)
        h = F.dropout(h, p=self._drop_p, training=self.training)
        h = self.tdnn2(h)
        h = F.dropout(h, p=self._drop_p, training=self.training)
        h = self.tdnn3(h)
        h = F.dropout(h, p=self._drop_p, training=self.training)
        h = self.tdnn4(h)
        h = F.dropout(h, p=self._drop_p, training=self.training)
        h = self.tdnn5(h)
        h = self.statspool(h)
        h = self.fc1(h) # [N, 512]
        h = F.dropout(h, p=self._drop_p, training=self.training)
        h = self.fc2(h)
        return self.affine(h)

    def extract_xvec(self, x, num=1):  # x [N, C, T]
        h = self.tdnn1(x)
        h = self.tdnn2(h)
        h = self.tdnn3(h)
        h = self.tdnn4(h)
        h = self.tdnn5(h)
        h = self.statspool(h)
        out_1 = self.fc1.fc(h)
        out_2 = self.fc2.fc(self.fc1(out_1))
        # out_2 = self.fc2.fc(self.fc1(h))
        return out_1 if num == 1 else out_2


if __name__ == "__main__":
    class test(nn.Module):
        def print_state(self):
            print(self.training)
    
    A = test()
    A.eval()
    A.print_state()
    A.train()
    A.print_state()
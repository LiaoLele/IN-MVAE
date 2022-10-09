import torch.nn as nn
import torch
from ..config.hparam import hparam as hp


class speaker_encoder_ge2e(nn.Module):

    """ speaker encoder network based on 3-layered lstm, followed by a projection layer to produce embeddings """
    """ Input:  x is spectrogram or melspectrogram of audio;   dimension is [batchsize, n_frame, d_feature]
        Output: h is the extracted embedding of speakers;      dimension is [batchsize, d_embedding]"""

    def __init__(self):
        super(speaker_encoder_ge2e, self).__init__()
        self.lstm = nn.LSTM(hp.model.nmels, hp.model.hidden, num_layers=hp.model.num_layer, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(hp.model.hidden, hp.model.proj)

    def extract_embd(self, x):   # x: [batchsize, time, d_feature]
        batchsize = x.shape[0]
        x = self.slide(x)
        h, _ = self.lstm(x)
        h = self.projection(h[:, -1, :])
        h = h.view(-1, batchsize, hp.model.proj)
        h = torch.mean(h, dim=0)
        h = h / torch.norm(h, dim=1, keepdim=True)
        assert (torch.norm(h, dim=1) - 1).abs().min() < 1e-6
        return h
        
    def slide(self, x):  # x: [batchsize, nfreq, time]
        frame_num = hp.train.spencoder_frame_num
        frame_hop = int(hp.train.spencoder_frame_hop * frame_num)
        sample_num = int((x.shape[-1] - frame_num) // frame_hop + 1)
        x_seg = []
        for i in range(sample_num):
            x_seg.append(x[:, :, i * frame_hop: i * frame_hop + frame_num])
        x_seg = torch.stack(x_seg, dim=0)
        x_seg = x_seg.view(-1, x_seg.shape[-2], x_seg.shape[-1])
        x_seg = x_seg.permute(0, 2, 1)
        assert x_seg.shape[1] == hp.train.spencoder_frame_num
        return x_seg
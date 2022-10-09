import torch.nn.functional as F
import torch.nn as nn
import torch


EPS = 1e-9


def pad_input(x, kernel_size, pad_type='reflect'):
    if kernel_size % 2 == 0:
        pad = (kernel_size // 2, kernel_size // 2 - 1)
    else:
        pad = (kernel_size // 2, kernel_size // 2)
    return F.pad(x, pad=pad, mode=pad_type)


def conv_banks_forward(x, conv_bank_layers, activation_layer):
    out = []
    for conv_bank in conv_bank_layers:
        y = conv_bank(pad_input(x, conv_bank.kernel_size[0]))
        out.append(activation_layer(y))
    out = out + [x]
    return torch.cat(out, dim=1)


class SpeakerEncoder(nn.Module):
    # TODO: batchnorm implementation

    def __init__(self, c_in, c_out, n_conv_blocks, n_dense_blocks,
                 conv_bank_scale, max_bank_width, c_bank,
                 c_h, kernel_size, stride_list,
                 activation_layer, dropout_rate, batchnorm):
        super(SpeakerEncoder, self).__init__()
        self._dropout_rate = dropout_rate
        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.kernel_size = kernel_size
        self.stride_list = stride_list

        self.act_layer = eval(activation_layer)
        self.dropout_layer = nn.Dropout(dropout_rate)

        self.conv_bank_layers = nn.ModuleList(
            [nn.Conv1d(self.c_in, c_bank, width) for width in range(conv_bank_scale, max_bank_width + 1, conv_bank_scale)]
        )
        n_conv_banks = int(max_bank_width // conv_bank_scale)
        c_in_next = self.c_in + c_bank * n_conv_banks

        self.bottle_neck_layer = nn.Conv1d(c_in_next, self.c_h, 1)
        self.convblock_first_layers = nn.ModuleList(
            [nn.Conv1d(c_h, c_h, kernel_size) for _ in range(self.n_conv_blocks)]
        )
        self.convblock_second_layers = nn.ModuleList(
            [nn.Conv1d(c_h, c_h, kernel_size, stride=stride_list[i]) for i in range(self.n_conv_blocks)]
        )
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.denseblock_first_layer = nn.ModuleList(
            [nn.Linear(c_h, c_h) for _ in range(self.n_dense_blocks)]
        )
        self.denseblock_second_layer = nn.ModuleList(
            [nn.Linear(c_h, c_h) for _ in range(self.n_dense_blocks)]
        )
        self.affine_out_layer = nn.Linear(c_h, c_out)

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, rate):
        if rate < 0.0 or rate > 1.0:
            raise ValueError("Invalid Dropout Rate!")
        self._dropout_rate = rate
        self.dropout_layer = nn.Dropout(self._dropout_rate)

    def conv_blocks_forward(self, x):
        out = x
        for block_idx in range(self.n_conv_blocks):
            y = self.convblock_first_layers[block_idx](pad_input(out, self.kernel_size))
            y = self.dropout_layer(self.act_layer(y))
            y = self.convblock_second_layers[block_idx](pad_input(y, self.kernel_size))
            y = self.dropout_layer(self.act_layer(y))
            if self.stride_list[block_idx] > 1:
                out = F.avg_pool1d(out, kernel_size=self.stride_list[block_idx], ceil_mode=True)
            out = out + y
        return out

    def dense_blocks_forward(self, x):
        out = x
        for block_idx in range(self.n_dense_blocks):
            y = self.denseblock_first_layer[block_idx](out)
            y = self.dropout_layer(self.act_layer(y))
            y = self.denseblock_second_layer[block_idx](y)
            y = self.dropout_layer(self.act_layer(y))
            out = out + y
        return out

    def forward(self, x):
        x = torch.log(x + EPS)
        out = conv_banks_forward(x, self.conv_bank_layers, self.act_layer)
        out = self.dropout_layer(self.act_layer(self.bottle_neck_layer(pad_input(out, self.bottle_neck_layer.kernel_size[0]))))  # official version里此处没有dropout
        out = self.conv_blocks_forward(out)
        out = self.global_pooling(out).squeeze(-1)
        out = self.dense_blocks_forward(out)
        out = self.affine_out_layer(out)
        return out


class ContentEncoder(nn.Module):
    
    def __init__(self, c_in, c_out, n_conv_blocks,
                 conv_bank_scale, max_bank_width, c_bank,
                 c_h, kernel_size, stride_list,
                 activation_layer, dropout_rate):
        super(ContentEncoder, self).__init__()
        self._dropout_rate = dropout_rate
        self.c_in = c_in
        self.c_out = c_out
        self.c_h = c_h
        self.n_conv_blocks = n_conv_blocks
        self.kernel_size = kernel_size
        self.stride_list = stride_list

        self.act_layer = eval(activation_layer)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.instancenorm_layer = nn.InstanceNorm1d(c_h, affine=False)

        self.input_layer = nn.Conv1d(c_in, c_h[0], 1)
        self.convblock_first_layers = nn.ModuleList(
            [nn.Conv1d(c_h[i], c_h[i + 1], kernel_size) for i in range(self.n_conv_blocks)]
        )
        self.convblock_second_layers = nn.ModuleList(
            [nn.Conv1d(c_h[i + 1], c_h[i + 1], kernel_size, stride=stride_list[i]) for i in range(self.n_conv_blocks)]
        )
        self.dimreduction_layers = nn.ModuleList(
            [nn.Conv1d(c_h[i], c_h[i + 1], 1) for i in range(self.n_conv_blocks)]
        )
        self.affine_mu_layer = nn.Conv1d(c_h[-1], c_out, 1)
        self.affine_logvar_layer = nn.Conv1d(c_h[-1], c_out, 1)

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, rate):
        if rate < 0.0 or rate > 1.0:
            raise ValueError("Invalid Dropout Rate!")
        self._dropout_rate = rate
        self.dropout_layer = nn.Dropout(self._dropout_rate)

    def conv_blocks_forward(self, x):
        out = x
        for block_idx in range(self.n_conv_blocks):
            y = self.convblock_first_layers[block_idx](pad_input(out, self.kernel_size))
            y = self.dropout_layer(self.act_layer(self.instancenorm_layer(y)))
            y = self.convblock_second_layers[block_idx](pad_input(y, self.kernel_size))
            y = self.dropout_layer(self.act_layer(self.instancenorm_layer(y)))
            out = self.dimreduction_layers[block_idx](pad_input(out, 1))
            if self.stride_list[block_idx] > 1:
                out = F.avg_pool1d(out, kernel_size=self.stride_list[block_idx], ceil_mode=True)
            out = out + y
        return out

    def forward(self, x):
        x = torch.log(x + EPS)
        out = self.input_layer(pad_input(x, self.input_layer.kernel_size[0]))
        out = self.dropout_layer(self.act_layer(self.instancenorm_layer(out)))
        out = self.conv_blocks_forward(out)
        mu = self.affine_mu_layer(pad_input(out, self.affine_mu_layer.kernel_size[0]))
        log_var = self.affine_logvar_layer(pad_input(out, self.affine_logvar_layer.kernel_size[0]))
        return mu, log_var


class Decoder(nn.Module):

    def __init__(self, c_cont, c_spkr, c_out, n_conv_blocks, 
                 c_h, kernel_size, upsamp_list,
                 activation_layer, dropout_rate):
        super(Decoder, self).__init__()
        self._dropout_rate = dropout_rate
        self.c_cont = c_cont
        self.c_spkr = c_spkr
        self.c_out = c_out
        self.c_h = c_h
        self.n_conv_blocks = n_conv_blocks
        self.kernel_size = kernel_size
        self.upsamp_list = upsamp_list

        self.act_layer = eval(activation_layer)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.instancenorm_layer = nn.InstanceNorm1d(c_h, affine=False)

        self.input_layer = nn.Conv1d(c_cont, c_h[0], 1)   # official version里考虑了谱归一化
        self.convblock_first_layers = nn.ModuleList(
            [nn.Conv1d(c_h[i], c_h[i + 1], kernel_size) for i in range(self.n_conv_blocks)]
        )
        self.convblock_second_layers = nn.ModuleList(
            [nn.Conv1d(c_h[i + 1], c_h[i + 1] * upsamp_list[i], kernel_size) for i in range(self.n_conv_blocks)]
        )
        self.dimexpansion_layers = nn.ModuleList(
            [nn.Conv1d(c_h[i], c_h[i + 1], 1) for i in range(self.n_conv_blocks)]
        )
        self.affine_spkrmean_layers = nn.ModuleList(
            [nn.Linear(c_spkr, c_h[i // 2 + 1]) for i in range(self.n_conv_blocks * 2)]
        )
        self.affine_spkrstd_layers = nn.ModuleList(
            [nn.Linear(c_spkr, c_h[i // 2 + 1]) for i in range(self.n_conv_blocks * 2)]
        )
        self.affine_out_layer = nn.Conv1d(c_h[-1], c_out, 1)

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, rate):
        if rate < 0.0 or rate > 1.0:
            raise ValueError("Invalid Dropout Rate!")
        self._dropout_rate = rate
        self.dropout_layer = nn.Dropout(self._dropout_rate)
        
    def pixelshuffle(self, x, upscale=2):
        batch_size, in_channels, in_frames = x.size()
        assert in_channels % upscale == 0
        out_channels = int(in_channels // upscale)
        out_frames = int(upscale * in_frames)
        x_view = x.contiguous().view(batch_size, out_channels, upscale, in_frames)
        out = x_view.permute(0, 1, 3, 2).contiguous()
        out = out.view(batch_size, out_channels, out_frames)
        return out

    def conv_blocks_forward(self, x, spkr_embd):
        out = x
        for block_idx in range(self.n_conv_blocks):
            y = self.instancenorm_layer(self.convblock_first_layers[block_idx](pad_input(out, self.kernel_size)))
            spkr_mean = self.affine_spkrmean_layers[block_idx * 2](spkr_embd)
            spkr_std = self.affine_spkrstd_layers[block_idx * 2](spkr_embd)
            y = self.dropout_layer(self.act_layer(y * spkr_std.unsqueeze(2) + spkr_mean.unsqueeze(2)))

            out = self.dimexpansion_layers[block_idx](pad_input(out, 1))

            y = self.convblock_second_layers[block_idx](pad_input(y, self.kernel_size))
            if self.upsamp_list[block_idx] > 1:
                y = self.pixelshuffle(y, upscale=self.upsamp_list[block_idx])
                out = F.interpolate(out, scale_factor=self.upsamp_list[block_idx], mode='nearest')
            y = self.instancenorm_layer(y)
            spkr_mean = self.affine_spkrmean_layers[block_idx * 2 + 1](spkr_embd)
            spkr_std = self.affine_spkrstd_layers[block_idx * 2 + 1](spkr_embd)
            y = self.dropout_layer(self.act_layer(y * spkr_std.unsqueeze(2) + spkr_mean.unsqueeze(2)))
            out = out + y
        return out

    def forward(self, cont_embd, spkr_embd):
        out = self.input_layer(pad_input(cont_embd, self.input_layer.kernel_size[0]))
        out = self.dropout_layer(self.act_layer(self.instancenorm_layer(out)))
        out = self.conv_blocks_forward(out, spkr_embd)
        out = self.affine_out_layer(pad_input(out, self.affine_out_layer.kernel_size[0]))
        return out


class AutoEncoder(nn.Module):

    def __init__(self, config):
        super(AutoEncoder, self).__init__()
        # self.log_factor = nn.Parameter(torch.ones([]))
        self.content_encoder = ContentEncoder(**config.ContentEncoder)
        self.speaker_encoder = SpeakerEncoder(**config.SpeakerEncoder)
        self.decoder = Decoder(**config.Decoder)

    # def forward(self, x_spectrogram_target, x_spectrogram_noisy):
    #     spkr_embd = self.speaker_encoder(x_spectrogram_target)
    #     content_mu, content_log_var = self.content_encoder(x_spectrogram_noisy)
    #     cont_embd = content_log_var.new_empty(content_log_var.size()).normal_(0, 1)
    #     cont_embd = cont_embd * torch.exp(content_log_var / 2) + content_mu
    #     # x_log_var = self.decoder(cont_embd, spkr_embd) + self.log_factor
    #     x_log_var = self.decoder(cont_embd, spkr_embd)
    #     return spkr_embd, content_mu, content_log_var, x_log_var

    def forward(self, x_spectrogram):
        spkr_embd = self.speaker_encoder(x_spectrogram)
        content_mu, content_log_var = self.content_encoder(x_spectrogram)
        cont_embd = content_log_var.new_empty(content_log_var.size()).normal_(0, 1)
        cont_embd = cont_embd * torch.exp(content_log_var / 2) + content_mu
        # x_log_var = self.decoder(cont_embd, spkr_embd) + self.log_factor
        x_log_var = self.decoder(cont_embd, spkr_embd)
        return spkr_embd, content_mu, content_log_var, x_log_var

    def inference(self, target_feat, source_feat, target_seglen=200, target_seghop=0.5):
        spkr_embd = self.get_speaker_embeddings(target_feat, target_seglen=target_seglen, target_seghop=target_seghop)
        src_mu, _ = self.content_encoder(source_feat)
        # x_log_var = self.decoder(src_mu, spkr_embd) + self.log_factor
        x_log_var = self.decoder(src_mu, spkr_embd)[:, :, : source_feat.shape[-1]]
        return (x_log_var * 0.5).exp()

    def get_speaker_embeddings(self, target_feat, target_seglen=200, target_seghop=0.5):
        if target_feat.dim() == 2:
            target_feat.unsqueeze(0)
        elif target_feat.dim() <= 1 or target_feat.dim() > 3:
            raise ValueError("Input without the correct dimension!")
        batch_size = target_feat.shape[0]
        target_seghop = int(target_seglen * target_seghop)
        seg_num = int((target_feat.shape[-1] - target_seglen) // target_seghop + 1)
        target_seg = []
        for i in range(seg_num):
            target_seg.append(target_feat[:, :, i * target_seghop: i * target_seghop + target_seglen])
        target_seg = torch.stack(target_seg, dim=0).view(-1, target_feat.shape[-2], target_seglen)
        spkr_embd = self.speaker_encoder(target_seg)
        spkr_embd = spkr_embd.view(-1, batch_size, spkr_embd.shape[-1])
        spkr_embd = torch.mean(spkr_embd, dim=0, keepdim=False)
        return spkr_embd


class Loss():
    def __init__(self, lambda_rec, lambda_kl, freq_weight=None, loss_type='basic', reduce_mode='sum'):
        self.lambda_rec = lambda_rec
        self.lambda_kl = lambda_kl
        self.loss_type = loss_type
        self.reduce_mode = reduce_mode
        if freq_weight is None:
            self.freq_weight = freq_weight
        else:
            self.freq_weight = freq_weight().squeeze()

    def __call__(self, oracle, content_mu, content_log_var, x_log_var):
        self.batch_size = oracle.shape[0]
        self.oracle = oracle
        self.content_mu = content_mu
        self.content_log_var = content_log_var
        self.x_log_var = x_log_var
        if self.freq_weight is not None:
            self.freq_weight = self.freq_weight.float().to(oracle.device)

        is_div = self.is_diff()
        L_kl = self.kl_loss()
        if self.loss_type == 'basic':
            L_rec = self.basic()
        elif self.loss_type == 'l1_loss':
            L_rec = self.l1_loss()
        elif self.loss_type == 'mse_loss':
            L_rec = self.mse_loss()
        loss = self.lambda_rec * L_rec + self.lambda_kl * L_kl
        # loss = loss * 1e-6
        return loss, is_div, L_rec

    def weighted_loss(self, L_rec):
        if self.reduce_mode == 'mean':
            L_rec = torch.mean(L_rec, axis=(0, 2))
            if self.freq_weight is not None:
                L_rec = self.freq_weight * L_rec
            L_rec = torch.mean(L_rec)
        else:
            L_rec = torch.sum(L_rec, axis=2)
            L_rec = torch.mean(L_rec, axis=0)
            if self.freq_weight is not None:
                L_rec = self.freq_weight * L_rec
            L_rec = torch.sum(L_rec)
        return L_rec

    def basic(self):
        L_rec = (self.oracle.log() - self.x_log_var).exp() + self.x_log_var
        L_rec = self.weighted_loss(L_rec)
        return L_rec

    def l1_loss(self):
        L_rec = F.l1_loss(self.x_log_var, (self.oracle + EPS).log(), reduction='none')
        L_rec = self.weighted_loss(L_rec)
        return L_rec

    def mse_loss(self):
        L_rec = F.mse_loss(self.x_log_var, (self.oracle + EPS).log(), reduction='none')
        L_rec = self.weighted_loss(L_rec)
        return L_rec

    def kl_loss(self):
        L_kl = 0.5 * torch.sum(self.content_mu.pow(2) + self.content_log_var.exp() - self.content_log_var)
        return L_kl / self.batch_size

    def is_diff(self):
        # is_div = torch.sum(self.oracle / self.x_log_var.exp() - (self.oracle.log() - self.x_log_var) - 1)
        # return is_div / self.batch_size
        is_div = torch.sum(self.oracle / (self.x_log_var.exp() + EPS) - ((self.oracle + EPS).log() - self.x_log_var) - 1, axis=(1, 2))
        is_div = torch.mean(is_div)
        return is_div


if __name__ == "__main__":
    from config import cfgs
    import numpy as np
    from torch.utils.data import DataLoader

    # device = torch.device('cpu')
    device = torch.device(1)
    model = AutoEncoder(cfgs).to(device)
    loss_ins = Loss(cfgs.loss.lambda_rec, cfgs.loss.lambda_kl, loss_type=cfgs.loss.loss_type)

    """ forward compare """
    # for param in model.parameters():
    #     param.data = 1e-3 * torch.ones_like(param.data)
    # np.random.seed(123)
    # spec_in = torch.from_numpy(np.random.rand(64, 513, 200).astype('float32')).to(device)
    # oracle = spec_in.clone()
    # # spec_in = torch.ones((1, 513, 200), device=device).float()
    # # oracle = torch.ones((1, 513, 200), device=device).float()

    # with torch.no_grad():
    #     spkr_embd, content_mu, content_log_var, x_log_var = model(spec_in, oracle)
    # spkr_embd_np, content_mu_np, content_log_var_np, x_log_var_np = spkr_embd.cpu().numpy().astype(np.float64), content_mu.cpu().numpy().astype(np.float64), content_log_var.cpu().numpy().astype(np.float64), x_log_var.cpu().numpy().astype(np.float64)
    # print("mean:{:.6e}, std:{:.6e}".format(np.mean(spkr_embd_np), np.std(spkr_embd_np)))
    # print("mean:{:.6e}, std:{:.6e}".format(np.mean(content_mu_np), np.std(content_mu_np)))
    # print("mean:{:.6e}, std:{:.6e}".format(np.mean(content_log_var_np), np.std(content_log_var_np)))
    # print("mean:{:.6e}, std:{:.6e}".format(np.mean(x_log_var_np), np.std(x_log_var_np)))
    # print('{:.10e}, {:.10e}'.format(np.max(spkr_embd_np), np.min(spkr_embd_np)))
    # print('{:.10e}, {:.10e}'.format(np.max(content_mu_np), np.min(content_mu_np)))
    # print('{:.10e}, {:.10e}'.format(np.max(content_log_var_np), np.min(content_log_var_np)))
    # print('{:.10e}, {:.10e}'.format(np.max(x_log_var_np), np.min(x_log_var_np)))
    # print(spkr_embd_np[0, :] - np.mean(spkr_embd_np))
    # loss, is_div = loss_ins(oracle, content_mu, content_log_var, x_log_var)
    # print(f"loss: {loss}, isdiv: {is_div}")
    # dummy = 1 

    """ Training compare """
    # for param in model.parameters():
    #     param.data = 1e-3 * torch.ones_like(param.data)
    # np.random.seed(123)
    # spec_in = torch.from_numpy(np.random.rand(64, 513, 200).astype('float32')).to(device)
    # dataloader = DataLoader(spec_in, batch_size=64, shuffle=False)
    # # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.0)
    # num_epoches = 8
    # for epoch in range(num_epoches):
    #     for i, data in enumerate(dataloader):
    #         spec_in = data.to(device).float()
    #         oracle = spec_in.clone()
    #         spkr_embd, content_mu, content_log_var, x_log_var = model(spec_in, oracle)
    #         loss, isdiv = loss_ins(oracle, content_mu, content_log_var, x_log_var)
    #         print(f"Epoch: {epoch}[{i}/{len(dataloader)}] --- loss: {loss}, isdiv: {isdiv}")
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    # dummy = 1

    """ Real data training compare """
    import copy
    from dataset import CleanDataset, TrainBatchSampler
    from PreProcess.data_utils import Spectrogram
    transform = Spectrogram(**cfgs.sigproc)
    def CollateFn(batch):
        spec_in = torch.stack(batch, 0).float()
        spec_in, _, _ = transform(spec_in)
        return spec_in, spec_in.clone()

    # for param in model.parameters():
    #     param.data = 1e-5 * torch.ones_like(param.data)
    train_dataset = CleanDataset(cfgs.path.train, cfgs.dataset.usecols_egs, cfgs.general.data_prefix)
    train_sampler = TrainBatchSampler.from_dataset(train_dataset, cfgs.dataloader.batch_size, n_batch=cfgs.dataloader.nbatch_train, drop_last=True, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=cfgs.dataloader.num_workers, collate_fn=CollateFn)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-10, momentum=0.0)
    num_epoches = 8
    for epoch in range(num_epoches):
        for i, (spec_in, oracle) in enumerate(train_loader):
            spec_in = spec_in.to(device)
            oracle = oracle.to(device)
            spkr_embd, content_mu, content_log_var, x_log_var = model(spec_in, oracle)
            loss, isdiv = loss_ins(oracle, content_mu, content_log_var, x_log_var)
            if i % 50 == 0 or i == 468:
                print(f"Epoch: {epoch}[{i}/{len(train_loader)}] --- loss: {loss}, isdiv: {isdiv}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    dummy = 1
    
    
    # for param in model.parameters():
    #     param.data = 1e-3 * torch.ones_like(param.data)
    # np.random.seed(123)
    # spec_in = torch.from_numpy(np.random.rand(64, 513, 200).astype('float32')).to(device)
    # dataloader = DataLoader(spec_in, batch_size=64, shuffle=False)
    # # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.0)
    # num_epoches = 8
    # for epoch in range(num_epoches):
    #     for i, data in enumerate(dataloader):
    #         spec_in = data.to(device).float()
    #         oracle = spec_in.clone()
    #         spkr_embd, content_mu, content_log_var, x_log_var = model(spec_in, oracle)
    #         loss, isdiv = loss_ins(oracle, content_mu, content_log_var, x_log_var)
    #         print(f"Epoch: {epoch}[{i}/{len(dataloader)}] --- loss: {loss}, isdiv: {isdiv}")
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    # dummy = 1

    # class test(nn.Module):
    #     def __init__(self, p):
    #         super(test, self).__init__()
    #         self._rate = p
    #         self.dropout_layer = nn.Dropout(p)

    #     @property
    #     def rate(self):
    #         return self._rate

    #     @rate.setter
    #     def rate(self, p):
    #         self._rate = p
    #         self.dropout_layer = nn.Dropout(self._rate)
    
    # mytest = test(0.2)
    # import ipdb; ipdb.set_trace()





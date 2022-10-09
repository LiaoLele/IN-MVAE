import torch.nn.functional as F
from torch.nn import Parameter
import torch
import math


EPS = 1e-6


class LossBase():
    def __init__(self, config, freq_weight=None):
        self.cfgs = config
        self.lambda_rec = self.cfgs.loss.lambda_rec
        self.lambda_kl = self.cfgs.loss.lambda_kl
        self.rec_loss_type = self.cfgs.loss.rec_loss_type
        self.rec_loss_reduce_mode = self.cfgs.loss.rec_loss_reduce_mode

        # multi-task losses
        self.lambda_classify = config.loss.lambda_classifyloss

        if freq_weight is None:
            self.freq_weight = freq_weight
        else:
            self.freq_weight = freq_weight().squeeze()

    def weighted_loss(self, L_rec):
        if self.rec_loss_reduce_mode == 'mean':
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

    def basic(self, oracle, x_log_var):
        L_rec = (oracle.clamp_min(EPS).log() - x_log_var).exp() + x_log_var
        L_rec = self.weighted_loss(L_rec)
        return L_rec

    def l1_loss(self, oracle, x_log_var):
        L_rec = F.l1_loss(x_log_var, (oracle + EPS).log(), reduction='none')
        L_rec = self.weighted_loss(L_rec)
        return L_rec

    def mse_loss(self, oracle, x_log_var):
        L_rec = F.mse_loss(x_log_var, (oracle + EPS).log(), reduction='none')
        L_rec = self.weighted_loss(L_rec)
        return L_rec

    def kl_loss(self, content_mu, content_log_var):
        L_kl = 0.5 * torch.sum(content_mu.pow(2) + content_log_var.exp() - content_log_var, dim=[1, 2])
        L_kl = torch.mean(L_kl)
        return L_kl

    def is_diff(self, oracle, x_log_var):
        is_div = torch.sum(oracle / (x_log_var.exp() + EPS) - ((oracle + EPS).log() - x_log_var) - 1, axis=(1, 2))
        is_div = torch.mean(is_div)
        return is_div


class Loss(LossBase):

    def __call__(self, oracle, label, *args):
        content_mu = args[0]
        content_log_var = args[1]
        x_log_var = args[2]
        if self.freq_weight is not None:
            self.freq_weight = self.freq_weight.float().to(oracle.device)

        # monitoring
        is_div = self.is_diff(oracle, x_log_var)

        # loss composition
        # kl loss
        L_kl = self.kl_loss(content_mu, content_log_var)
        # rec_loss
        if self.rec_loss_type == 'basic':
            L_rec = self.basic(oracle, x_log_var)
        elif self.rec_loss_type == 'l1':
            L_rec = self.l1_rec_loss(oracle, x_log_var)
        elif self.rec_loss_type == 'mse':
            L_rec = self.mse_loss(oracle, x_log_var)
        # classify loss
        if self.cfgs.loss.include_classification_loss:
            prob = args[3]
            L_clsy = F.cross_entropy(prob, label, reduction='mean')
        else:
            L_clsy = torch.tensor(0.0)

        # total_loss
        loss = self.lambda_rec * L_rec + self.lambda_kl * L_kl + self.lambda_classify * L_clsy
        return loss, is_div, L_rec, L_kl, L_clsy

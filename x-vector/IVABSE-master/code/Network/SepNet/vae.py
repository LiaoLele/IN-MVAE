import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GatedConvBN1d(torch.nn.Module):
    """1-D Gated convolution layer with batch normalization.
    Arguments are same as `torch.nn.Conv1d`.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(GatedConvBN1d, self).__init__()

        self.conv = torch.nn.Conv1d(
            in_channels, 2 * out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode)
        self.bn = torch.nn.BatchNorm1d(2 * out_channels)

    def forward(self, x):
        return F.glu(self.bn(self.conv(x)), dim=1)


class GatedDeconvBN1d(torch.nn.Module):
    """1-D Gated deconvolution layer with batch normalization.
    Arguments are same as `torch.nn.ConvTranspose1d`.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        super(GatedDeconvBN1d, self).__init__()

        self.deconv = torch.nn.ConvTranspose1d(
            in_channels, 2 * out_channels, kernel_size, stride, padding,
            output_padding, groups, bias, dilation, padding_mode)
        self.bn = torch.nn.BatchNorm1d(2 * out_channels)

    def forward(self, x):
        return F.glu(self.bn(self.deconv(x)), dim=1)
       

class net(torch.nn.Module):
    """Conditional VAE (M2 model) for MCVAE.

    Args:
        n_embedding (int): Number of embedding.

    Input: x_stft [B*3, n_freq, n_time]

    Output: 
    """
    def __init__(self, n_embedding):
        super(net, self).__init__()

        self.n_embedding = n_embedding
        self.eps = 1e-6
        self.log_g = Parameter(torch.ones([]))

        self.encoder_conv1 = GatedConvBN1d(
            513 + self.n_embedding, 256, kernel_size=5, stride=1, padding=2)
        self.encoder_conv2 = GatedConvBN1d(
            256 + self.n_embedding, 128, kernel_size=4, stride=2, padding=1)
        self.encoder_mu = torch.nn.Conv1d(
            128 + self.n_embedding, 64, kernel_size=4, stride=2, padding=1)
        self.encoder_logvar = torch.nn.Conv1d(
            128 + self.n_embedding, 64, kernel_size=4, stride=2, padding=1)

        self.decoder_deconv1 = GatedDeconvBN1d(
            64 + self.n_embedding, 128, kernel_size=4, stride=2, padding=1)
        self.decoder_deconv2 = GatedDeconvBN1d(
            128 + self.n_embedding, 256, kernel_size=4, stride=2, padding=1)
        self.decoder_deconv3 = torch.nn.ConvTranspose1d(
            256 + self.n_embedding, 513, kernel_size=5, stride=1, padding=2)

    @property
    def d_embedding(self):
        return self.n_embedding

    def encode(self, x, c):
        x = torch.log(x + self.eps)
        c = c.unsqueeze(2)
        h = torch.cat((x, c.expand(-1, -1, x.size(2))), dim=1)
        h = self.encoder_conv1(h)
        h = torch.cat((h, c.expand(-1, -1, h.size(2))), dim=1)
        h = self.encoder_conv2(h)
        h = torch.cat((h, c.expand(-1, -1, h.size(2))), dim=1)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar

    def decode(self, z, c):
        c = c.unsqueeze(2)
        h = torch.cat((z, c.expand(-1, -1, z.size(2))), dim=1)
        h = self.decoder_deconv1(h)
        h = torch.cat((h, c.expand(-1, -1, h.size(2))), dim=1)
        h = self.decoder_deconv2(h)
        h = torch.cat((h, c.expand(-1, -1, h.size(2))), dim=1)
        log_sigma_sq = self.decoder_deconv3(h)
        return log_sigma_sq
    
    def forward(self, x_spec, label=None):  
        pass
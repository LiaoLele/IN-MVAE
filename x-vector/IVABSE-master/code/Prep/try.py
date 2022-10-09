import scipy.io as sio
import soundfile as sf
import numpy as np
import librosa as rosa
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.realpath(os.path.dirname(__file__)))))
from Network.SepNet import vae
# from SepAlgo.ilrma import myilrma, ilrma
# from SepAlgo.gmm import avgmm
from SepAlgo.Aux_chainlike import chainlike, chainlike_prob
from SepAlgo.mvae import mvae_onehot, mvae_ge2e
from utils import mysort, zero_pad, assign_rir, assign_sir
from pyroomacoustics.bss.ilrma import ilrma

import torch

# prefix = '/home/user/zhaoyi.gu/mnt/g2/'
# datapath = prefix + 'DATASET/MIRD/IR_0.16_8-8-8-8-8-8-8/Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.160s)_8-8-8-8-8-8-8_1m_090.mat'
data = sio.loadmat(datapath)
# data_from_matlab_file = prefix + 'DATASET/MIRD/test.wav'
# data_from_matlab, fs = sf.read(data_from_matlab_file) 
# data_from_matlab_2, fs_2 = librosa.load(data_from_matlab_file, sr=None, mono=False, dtype=np.float64)
# dummy = 1

# path = '/data/hdd0/zhaoyigu/DATASET/wpe_ilrma/'
# os.makedirs(os.path.join(path, 'sep'), exist_ok=True)
# file = list(filter(lambda x: x.endswith('.wav'), os.listdir(path)))
# nfft = 1024
# hop = 256
# fs = 8000
# for f in file:
#     mix, _ = sf.read(os.path.join(path, f))
#     mix_spec = [librosa.core.stft(np.asfortranarray(mix[:, ch]), n_fft=nfft, hop_length=hop, win_length=nfft) for ch in range(mix.shape[1])]
#     mix_spec = np.stack(mix_spec, axis=1)
#     sep_spec = ilrma(mix_spec.transpose(2, 0, 1), n_iter=200, n_components=2)
#     sep_spec = sep_spec.transpose(1, 2, 0)
#     # sep_spec, flag = myilrma(mix_spec, 1000, n_basis=2)
#     sep = [librosa.core.istft(sep_spec[:, ch, :], hop_length=hop, length=mix.shape[0]) for ch in range(sep_spec.shape[1])]
#     sep = np.stack(sep, axis=1)
#     sf.write(os.path.join(path, 'sep', f[0: -4] + '_sep.wav'), sep, fs)



if __name__ == "__main__":
    mix_wav_path = '/data/hdd0/zhaoyigu/DATASET/tmp/male_female_aec.wav'
    output_path = '/data/hdd0/zhaoyigu/DATASET/tmp/mvae/sep_chainlike_4ch.wav'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # vae_model_path = '/data/hdd0/zhaoyigu/PROJECT/MVAE_w_embds_data/output/onehot_embds/test_librispeech_500sp_15min/model/state_dict--epoch=2000.pt'
    # modelfile_path = vae_model_path
    # n_embedding = 500
    # device = torch.device(0)
    # model = vae.net(n_embedding)
    # model.load_state_dict(torch.load(modelfile_path, map_location=device))
    # model.to(device)
    # model.eval()

    audio, _ = rosa.load(mix_wav_path, sr=16000, mono=False)
    # audio = audio[0:2,:]
    # audio = zero_pad(audio, 4, hop_length=256)
    # src, _ = rosa.load(src_wav_path, sr=16000, mono=False)
    mix_spec = np.stack([rosa.stft(np.asfortranarray(x_c), n_fft=1024, hop_length=256) for x_c in audio], axis=1)
    # sep_spec, flag = chainlike(mix_spec, clique_bins=128, clique_hop=1)
    sep_spec, flag, time_all, 
    sep = np.stack([rosa.istft(sep_spec[:, ch, :], hop_length=256) for ch in range(0, 4)], axis=0)
    sf.write(output_path, sep.T, 16000)
    


class online_mvae(online_iva):

    def add_par(self, vae_model_path=None, device=None, d_embed=None, block_loop_num=1, sgd_loop_num=10):
        self.device = device
        self.d_embed = d_embed
        self.model = vae.net(d_embed)
        self.model.load_state_dict(torch.load(vae_model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.block_loop_num = block_loop_num
        self.sgd_loop_num = sgd_loop_num

    def para_init(self):
        self.sep_mat = np.stack([np.eye(self.nch, dtype=np.complex64) for _ in range(self.n_freq)], axis=0)  # [n_freq, nch, nch]
        self.eye = self.sep_mat.copy()
        self.sep_mat_tmp = self.sep_mat.copy()
        self.V = np.zeros((self.nch, self.n_freq, self.nch, self.nch), dtype=np.complex64)  # [self.nch, n_freq, self.nch, self.nch]
        self.V_tmp = self.V.copy()
        if self.coeff_type == 1:
            self.coeff_buffer = 0
        if self.use_fast_W:
            self.V_inv = np.ones((self.nch, self.n_freq, self.nch, self.nch), dtype=np.complex64)
            self.sep_mat_inv = self.sep_mat.copy()
            self.V_inv_tmp = self.V_inv.copy()

        self.log_g = torch.full((self.nch, 1, 1), self.model.log_g.item(), device=self.device)
        self.c = 1 / self.d_embed * torch.ones((self.nch, self.d_embed), device=self.device, requires_grad=True)
        self.z = torch.randn((self.nch, self.model.encoder_mu.out_channels, 2), device=self.device, requires_grad=True)
        self.out_idx = 0

    def one_iteration(self, stft_buffer, iter_idx):
        batch_len = stft_buffer.shape[-1]
        self.out_idx += 1
        alpha = self._get_alpha()
        if iter_idx % 4 == 0:  # 添加新的z
            self.z = torch.cat(
                (self.z.data, torch.randn((self.nch, self.model.encoder_mu.out_channels, 1), device=self.device)), dim=2)
            self.z.requires_grad = True
        if iter_idx >= 9 and iter_idx % 4 == 2:  # 去掉没有用的z以减少计算量
            self.z.data = self.z.data[:, :, 1:]
            self.z.requires_grad = True
            self.out_idx = self.out_idx - 4
        for block_loop in range(self.block_loop_num):
            # 得到分离信号
            sep = self.sep_mat @ stft_buffer
            sep_abs = np.abs(sep)
            sep_pow = sep_abs ** 2  # [n_freq, self.nch, batch_len]
            # 更新z, c
            assert self.z.is_leaf
            assert self.z.requires_grad
            assert self.c.requires_grad
            sep_pow_tensor = torch.from_numpy(sep_pow.swapaxes(0, 1)).to(self.device)
            optimizer = torch.optim.Adam((self.z, self.c), lr=1e-3)
            for sgd_loop in range(self.sgd_loop_num):
                log_sigma = self.model.decode(self.z, torch.softmax(self.c, dim=1)) + self.log_g
                log_sigma = log_sigma[:, :, self.out_idx: self.out_idx + 1]
                loss = torch.sum(log_sigma + (sep_pow_tensor.log() - log_sigma).exp())
                self.model.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # 更新 log_g
            with torch.no_grad():
                sigma = (self.model.decode(self.z, torch.softmax(self.c, dim=1))).exp()
                sigma = sigma[:, :, self.out_idx: self.out_idx + 1]
                lbd = torch.sum(sep_pow_tensor / sigma / self.n_freq, dim=(1, 2))
                # gt update
                # self.log_g[:, 0, 0] = torch.log(lbd)
                # g update
                self.log_g[:, 0, 0] = torch.log(alpha * self.log_g[:, 0, 0]**2 + (1 - alpha) * lbd)
                sigma = (self.model.decode(self.z, torch.softmax(self.c, dim=1)) + self.log_g).exp()
                reci = (1 / (sigma + 1e-7)).cpu().numpy()
            # 更新 sep_mat
            if self.use_fast_W:   # only used in fully online version
                for src in range(self.nch):
                    pass
            else:
                for src in range(self.nch):  # only used in block batch version
                    V_current = (stft_buffer * reci[src, :, None, :]) @ stft_buffer.conj().swapaxes(1, 2)
                    V_current = V_current / batch_len
                    self.V_tmp[src, :, :, :] = alpha * self.V[src, :, :, :] + (1 - alpha) * V_current  # V [nsrc, nfreq, self.nch, self.nch]
                    if iter_idx == 0 and batch_len == 1:
                        self.V_tmp[src, :, :, :] += 1e-3 * np.min(np.abs(self.V_tmp[src, :, :, :]), axis=(1, 2), keepdims=True) * self.eye
                    U_mat = self.sep_mat @ self.V_tmp[src, :, :, :]
                    inv_WV = np.linalg.inv(U_mat + 0.001 * np.min(np.abs(U_mat), axis=(1, 2), keepdims=True) * self.eye)  #
                    self.sep_mat_tmp[:, :, src] = inv_WV[:, :, src].copy()
                    norm_fac = self.sep_mat_tmp[:, None, :, src].conj() @ self.V_tmp[src, :, :, :] @ self.sep_mat_tmp[:, :, src, None] + 1e-8
                    self.sep_mat_tmp[:, :, src] = self.sep_mat_tmp[:, :, src] / np.sqrt(norm_fac.squeeze(2))
                self.sep_mat = self.sep_mat_tmp.conj().swapaxes(1, 2)
        # update cumulative values
        if self.coeff_type == 1:
            self.coeff_buffer = self.coeff * self.coeff_buffer + 1
        self.V = self.V_tmp.copy()
        # scale compensation
        for k in range(self.n_freq):
            sep[k, :, :] = np.diag(np.diag(np.linalg.inv(self.sep_mat[k, :, :]))) @ self.sep_mat[k, :, :] @ stft_buffer[k, :, :]  # [n_freq, self.nch, batch_len]
        return sep

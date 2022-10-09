import torch
import librosa
import torchaudio
import numpy as np
from config import cfgs
import copy

device = cfgs.general.device

def _mel_to_linear_matrix(sr, n_fft, n_mels):
    m = torch.from_numpy(librosa.filters.mel(sr, n_fft, n_mels)).to(device)
    m_t = m.transpose(0,1)
    p = torch.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p.detach().cpu().numpy(), axis=0)]
    return torch.matmul(m_t, torch.from_numpy(np.diag(d)).float().to(device))


def invert_spectrogram(spectrogram, src_len):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hop_length=int(cfgs.sigproc.sr*cfgs.sigproc.hop_len), length=src_len, win_length=int(cfgs.sigproc.sr*cfgs.sigproc.stft_len), window="hann")


def griffin_lim(spectrogram, src_len):
    '''Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    # X_t = librosa.core.istft(spectrogram, hop_length=int(cfgs.sigproc.sr*cfgs.sigproc.hop_len), length=src_len)
    for i in range(100):
        X_t = invert_spectrogram(X_best, src_len)
        est = librosa.stft(X_t, int(cfgs.sigproc.sr*cfgs.sigproc.stft_len), hop_length=int(cfgs.sigproc.sr*cfgs.sigproc.hop_len), win_length=int(cfgs.sigproc.sr*cfgs.sigproc.stft_len))
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
        # X_t = invert_spectrogram(X_best, src_len)
    # print('stft_len')
    # print(int(cfgs.sigproc.sr*cfgs.sigproc.stft_len))
    # print('est.shape')
    # print(est.shape)
    X_t = invert_spectrogram(X_best, src_len)
    y = np.real(X_t)

    return y

def melspectrogram2wav(mel, src_len):
    '''# Generate wave file from spectrogram'''
    # dec = dec.transpose(1, 2).squeeze(0)

    # transpose
    # mel = mel.T

    # de-noramlize
    mel = (torch.clamp(mel, 0, 1) * 100) - 100 + 20

    # to amplitude
    mel = torch.pow(10.0, mel * 0.05)
    m = _mel_to_linear_matrix(cfgs.sigproc.sr, int(cfgs.sigproc.sr*cfgs.sigproc.stft_len), cfgs.sigproc.nmels)
    mag = torch.matmul(m, mel).detach().cpu().numpy()
    # print(mag.shape)

    # wav reconstruction
    wav = griffin_lim(mag, src_len)
    # wav[0] = 0
    # wav[-1] = 0
    wav = wav/np.max(np.abs(wav))
    # wav = librosa.core.istft(mag, hop_length=hop_len, length=src_len)

    # # de-preemphasis
    # wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    # wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)



class Spectrogram():
    def __init__(self, calc_mfccs=False,
                 calc_spec=True, spec_mode='mag',
                 calc_mel=False, mel_mode='db',
                 sr=16000, stft_len=0.064, hop_len=0.016, nmels=30, nmfcc=30,
                 eps=1e-5):
        self.spec, self.mel, self.mfcc = None, None, None
        self.calc_mfccs = calc_mfccs
        self.calc_spec = calc_spec
        self.spec_mode = spec_mode
        self.calc_mel = calc_mel
        self.mel_mode = mel_mode
        self.sr = sr
        self.nfft = int(sr * stft_len)
        self.hop = int(sr * hop_len)
        self.nmels = nmels
        self.nmfcc = nmfcc
        self.eps = eps

    def __call__(self, wav):
        device = wav.device
        wav = wav.float()
        with torch.no_grad():
            if self.calc_spec or self.calc_mel:
                mag = torch.stft(wav, self.nfft, win_length=self.nfft, hop_length=self.hop,
                                 window=torch.hann_window(self.nfft, device=device))
                # torch.stft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, 
                #            pad_mode='reflect', normalized=False, onesided=None, return_complex=None)
                # input must be either a 1-D time sequence or a 2-D batch of time sequences.
                # Returns either a complex tensor of size (∗×N×T) if return_complex is true, 
                # or a real tensor of size (∗×N×T×2). 
                # Where ∗ is the optional batch size of input, N is the number of frequencies and T is the total number of frames
                # return_complex is True (default if input is complex)但这里input是实数
                mag = mag.pow(2).sum(-1).pow(0.5)  # [batchsize, nfreq, T]
                if self.calc_spec:
                    self.spec = torch.clamp_min(mag, self.eps).float().to(device)
                    if self.spec_mode == 'db':
                        self.spec = 20 * torch.log10(self.spec)
                    if self.spec_mode == 'power':
                        self.spec = self.spec.pow(2)
                if self.calc_mel:
                    mel_basis = torch.from_numpy(librosa.filters.mel(self.sr, self.nfft, n_mels=self.nmels)).unsqueeze(0)
                    # return: np.ndarray [shape=(n_mels, 1 + n_fft/2)]
                    # self.mel = torch.matmul(mel_basis.cuda(device), mag).clamp_min_(self.eps).float().cuda(device)
                    self.mel = torch.matmul(mel_basis.to(device), mag).clamp_min_(self.eps).float().to(device)
                    # [batchsize, n_mels, T]
                    if self.mel_mode == 'db':
                        self.mel = 20 * torch.log10(self.mel)
                        # normalize
                        self.mel = torch.clamp((self.mel - 20 + 100) / 100, 1e-8, 1)
            if self.calc_mfccs:
                melkwargs = {
                    'sample_rate': self.sr,
                    'n_fft': self.nfft,
                    'hop_length': self.hop,
                    'n_mels': self.nmels,
                    'window_fn': torch.hann_window,
                }
                transform = torchaudio.transforms.MFCC(sample_rate=self.sr, n_mfcc=self.nmfcc, melkwargs=melkwargs)
                self.mfcc = transform(wav).float().cuda(device)

        return self.spec, self.mel, self.mfcc


def spectrogram2wav(mag):
    '''# Generate wave file from spectrogram'''
    wav = griffin_lim(mag)
    # wav, _ = librosa.effects.trim(wav)
    return wav.astype(np.float32)
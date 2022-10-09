import os
import torch
import librosa
import mir_eval
import numpy as np
import soundfile as sf
from mvae import mvae
from BSSAlgorithm.wav_reader import WavReader
# from BSSAlgorithm.cvae import CVAE
from cvae import CVAE

class Separation(object):  
    def __init__(self, model_path, nfft=4096, hopsize=None, device=torch.device(1), n_iter=100):
        self.model_path = model_path
        self.nfft = nfft
        self.hopsize = nfft // 2 if hopsize is None else hopsize
        self.device = device
        self.n_iter = n_iter
        self.model = torch.load(model_path)
        self.model.eval()
        self.model.to(device)

    def stft(self, data):
      
        X1 = librosa.stft(data[:,0].copy(), self.nfft, self.hopsize)
        X2 = librosa.stft(data[:,1].copy(), self.nfft, self.hopsize)
        X = np.stack((X1, X2), axis=1)
        return X

    def separate(self, mix_spec):
        
        separated, _ = mvae(mix_spec, self.model, n_iter=self.n_iter, device=self.device)
        # separated = [librosa.istft(separated[:, ch, :], 2048)
        #              for ch in range(separated.shape[1])]
        # separated = np.stack(separated, axis=0)
        
        return separated

    def run(self, s_mix, s_src):
        len_src = s_mix.shape[0]
        def zero_pad(x):
            stft_pad_len = x.shape[0] + self.nfft
            if (stft_pad_len - self.nfft) // self.hopsize % 2:
                return x
            pad_len = self.hopsize - stft_pad_len % self.hopsize
            return np.pad(x, ((0, pad_len), (0, 0)), mode='constant')

        s_mix = zero_pad(s_mix)
        mix_spec = self.stft(s_mix)
        s_sep = self.separate(mix_spec)[:,0:len_src]
        
        sdr, sir, sar, _ =\
            mir_eval.separation.bss_eval_sources(s_src, s_sep)
        obj_score = {"sdr": sdr, "sir": sir, "sar": sar}
        return s_sep, obj_score

if __name__ == "__main__":
    if_use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:{}'.format(0) if if_use_cuda else 'cpu')
    # device = torch.device("cpu")
    separator = Separation("/data/hdd0/leleliao/dnn/vae/model/mvae.pt",
                           device=device)
    reader =  WavReader('/home/nis/lele.liao/projects/FastIVE/fig2_mixture_300ms/10')
    ref_path = "/home/nis/lele.liao/projects/FastIVE/singlechannel"
    performance = {"sdr": [], "sir": [], "sar": []}
    idx = np.random.choice(len(reader), size=100, replace=False)
    for i in idx:
        filename = reader.filename(i)
        s_mix, fs = reader[i]
        names = os.path.split(filename)[1].split("-")
        s1, fs_loc = sf.read(os.path.join(ref_path,names[0]+'.wav'))
        s2, fs_loc = sf.read(os.path.join(ref_path,names[1]+'.wav'))
        s_src = np.stack((s1,s2), axis=0)

        s_sep, obj_score = separator.run(s_mix, s_src)
        for key in obj_score:
            performance[key].append(obj_score[key])
    
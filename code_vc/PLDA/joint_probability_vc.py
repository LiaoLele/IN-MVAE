import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.realpath(os.path.dirname(__file__)))))
import pickle
import numpy as np
import torchaudio
import torch
import copy
from BSSAlgorithm.ilrma import myilrma
from PreProcess.data_utils import _mel_to_linear_matrix
from config import cfgs
import librosa
import statistics as stat
import torch.nn.functional as F
import bob.learn.em
import bob.learn.linear
from xvector.hparam import hparam as hp
from PreProcess.data_utils import Spectrogram


def probability_vc(enroll_sig, sep_sigs, model, xvecmean_path, lda_path, plda_path):
    device = enroll_sig.device
    transform = Spectrogram(**cfgs.sigproc)
    model = model.eval()
    sep_sigs = torch.from_numpy(sep_sigs).to(device).float()

    enroll_sig = (enroll_sig - enroll_sig.mean(dim=-1, keepdim=True)) / (enroll_sig.std(dim=-1, keepdim=True))
    sep_sigs = (sep_sigs - sep_sigs.mean(dim=-1, keepdim=True)) / (sep_sigs.std(dim=-1, keepdim=True))

    # obtain spkr embedding
    with torch.no_grad():
        _, sep_spec_mel, _ = transform(sep_sigs)
        vec_sep = model.get_speaker_embeddings(sep_spec_mel)
        _, enroll_spec_mel, _ = transform(enroll_sig)
        vec_enroll = model.get_speaker_embeddings(enroll_spec_mel)
    vec_sep = vec_sep.cpu().numpy()
    vec_enroll = vec_enroll.cpu().numpy()

    # load machine
    with open(xvecmean_path, 'rb') as f:
        embd_mean = pickle.load(f)
    machine_file = bob.io.base.HDF5File(lda_path)
    lda_machine = bob.learn.linear.Machine(machine_file)
    del machine_file
    plda_hdf5 = bob.io.base.HDF5File(plda_path)
    plda_base = bob.learn.em.PLDABase(plda_hdf5)

    #LDA
    vec_sep = lda_machine.forward(vec_sep - embd_mean)
    vec_enroll = lda_machine.forward(vec_enroll - embd_mean)
    # vec_sep = vec_sep.astype(np.float64)
    # vec_enroll = vec_enroll.astype(np.float64)
    # vec_sep = lda_machine.forward(vec_sep)
    # vec_enroll = lda_machine.forward(vec_enroll)
    # vec_sep = vec_sep - embd_mean
    # vec_enroll = vec_enroll - embd_mean
    
    #length_norm
    vec_sep = vec_sep / np.linalg.norm(vec_sep, axis=1, keepdims=True)
    vec_enroll = vec_enroll / np.linalg.norm(vec_sep, axis=1, keepdims=True)
    vec_sep = vec_sep.astype(np.float64)
    vec_enroll = vec_enroll.astype(np.float64)
    
    # PLDA
    plda_machine_1 = bob.learn.em.PLDAMachine(plda_base)
    plda_machine_2 = bob.learn.em.PLDAMachine(plda_base)
    plda_trainer = bob.learn.em.PLDATrainer()
    plda_trainer.enroll(plda_machine_1, vec_enroll[0, None, :])
    plda_trainer.enroll(plda_machine_2, vec_enroll[1, None, :])
    loglike_enroll_1 = np.stack([plda_machine_1.compute_log_likelihood(vec_sep[0, :]), plda_machine_1.compute_log_likelihood(vec_sep[1, :])])
    loglike_enroll_2 = np.stack([plda_machine_2.compute_log_likelihood(vec_sep[0, :]), plda_machine_2.compute_log_likelihood(vec_sep[1, :])])
    target_idx_enroll_1 = loglike_enroll_1.argmax()#argmax(f(x)）就是使f(x)值最大的那个自变量x的值
    print(f"probability1:{loglike_enroll_1}, index:{target_idx_enroll_1}")
    target_idx_enroll_2 = loglike_enroll_2.argmax()
    print(f"probability2:{loglike_enroll_2}, index:{target_idx_enroll_2}")
    if target_idx_enroll_1 == target_idx_enroll_2:
        if loglike_enroll_1[target_idx_enroll_1] > loglike_enroll_2[target_idx_enroll_2]:
            if target_idx_enroll_1 == 1:
                sep_sigs = sep_sigs[[1, 0], :]
        else:
            if target_idx_enroll_2 == 0:
                sep_sigs = sep_sigs[[1, 0], :]
        # raise ValueError("contradiction!")
    else:
        sep_sigs = sep_sigs[[target_idx_enroll_1, target_idx_enroll_2], :]
    sep_sigs = sep_sigs.detach().cpu().numpy()
    return sep_sigs

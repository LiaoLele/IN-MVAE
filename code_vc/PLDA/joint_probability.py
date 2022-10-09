import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.realpath(os.path.dirname(__file__)))))
import pickle
import numpy as np
import torchaudio
import torch
import copy
import statistics as stat
import torch.nn.functional as F
import bob.learn.em
import bob.learn.linear
from xvector.hparam import hparam as hp
from xvector.xvec import speaker_encoder_xvec


def probability(enroll_sig, sep_sigs, xvecmodel_path, xvecmean_path, lda_path, plda_path):
    device = enroll_sig.device
    enroll_sig = enroll_sig.to('cpu')

    # load machine
    with open(xvecmean_path, 'rb') as f:
        embd_mean = pickle.load(f)
    machine_file = bob.io.base.HDF5File(lda_path)
    lda_machine = bob.learn.linear.Machine(machine_file)
    del machine_file
    # whitening_matrix = None
    plda_hdf5 = bob.io.base.HDF5File(plda_path)
    plda_base = bob.learn.em.PLDABase(plda_hdf5)

    #MFCC
    n_fft = int(hp.data.stft_frame * hp.data.sr)
    hop = int(hp.data.stft_hop * hp.data.sr)
    feat_num = int(hp.model.feat_num)
    fs = int(hp.data.sr)
    melsetting = {}
    melsetting['n_fft'] = n_fft
    melsetting['win_length'] = n_fft
    melsetting['hop_length'] = hop
    melsetting['n_mels'] = feat_num
    transform = torchaudio.transforms.MFCC(sample_rate=fs, n_mfcc=feat_num, melkwargs=melsetting)

    # Load speaker model
    spkr_model = speaker_encoder_xvec()
    spkr_model_dict = spkr_model.state_dict()
    pretrained_dict = torch.load(xvecmodel_path, map_location=torch.device('cpu'))
    pretrained_dict_rename = pretrained_dict
    pretrained_dict_rename = {k: v for k, v in pretrained_dict_rename.items() if k in spkr_model_dict} 
    spkr_model_dict.update(pretrained_dict_rename)
    spkr_model.load_state_dict(spkr_model_dict)
    spkr_model.cuda(device)
    spkr_model.eval()

    # obtain signals' x-vector
    with torch.no_grad():
        sep = torch.from_numpy(sep_sigs)
        sep = transform(sep.float())
        sep = sep.float().cuda(device)
        sep = (sep - sep.mean(dim=-1, keepdim=True)) / (sep.std(dim=-1, keepdim=True))
        vec_sep = spkr_model.extract_embd(sep, use_slide=True)
        enroll = transform(enroll_sig.float())
        enroll = enroll.float().cuda(device)
        enroll = (enroll - enroll.mean(dim=-1, keepdim=True)) / (enroll.std(dim=-1, keepdim=True))
        vec_enroll = spkr_model.extract_embd(enroll, use_slide=True)
    vec_sep = vec_sep.cpu().numpy()
    vec_enroll = vec_enroll.cpu().numpy()

    #LDA
    vec_sep = lda_machine.forward(vec_sep - embd_mean)
    vec_enroll = lda_machine.forward(vec_enroll - embd_mean)
    
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
    target_idx_enroll_1 = loglike_enroll_1.argmax()
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
    return sep_sigs

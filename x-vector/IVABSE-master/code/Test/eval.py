import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.realpath(os.path.dirname(__file__)))))
import pickle
import soundfile as sf
from itertools import product
import numpy as np
import mir_eval
import librosa
import torchaudio
import torch
import copy
import statistics as stat
import torch.nn.functional as F
import bob.learn.em
import bob.learn.linear
from pkg.config.hparam_eval import hparam as hp
from Network.SpkrNet.ge2e import speaker_encoder_ge2e
from Network.SpkrNet.xvec import speaker_encoder_xvec
from pkg.utils import my_spectrogram, spectrogram_normalize, zero_pad


n_fft = int(hp.data.stft_frame * hp.data.sr)
hop = int(hp.data.stft_hop * hp.data.sr)


def get_extracted_metrics(datafile_list, sep_method, suffix='0', rkg_method_list=None, fs=16000, sub_state=0, sep_suffix=''):

    if sep_suffix:
        src_len = int(int(sep_suffix) * fs)
        sep_suffix = str(sep_suffix) + 's.'
    else:
        src_len = int(int(30) * fs)
    for rkg_method in rkg_method_list:
        print(sep_method + '\n')
        print(rkg_method)
        print(sep_suffix)

        # recognize one by one
        for datafile in datafile_list:
            print(datafile)
            with open(datafile, 'rb') as f:
                makemix = pickle.load(f)  
            dataname = os.path.basename(datafile)[0: -4]   # e.g. makemix_same_00
            target_path = os.path.join(os.path.dirname(datafile), sep_method)  # path where separated .wav is saved

            sep_file = os.path.join(target_path, 'metrics-{}.0-120.'.format(dataname) + sep_suffix + 'pkl')
            with open(sep_file, 'rb') as f:
                sep_ret = pickle.load(f)

            rkgfile = os.path.join(target_path, "rkg_ret-{}-{}.0.".format(dataname, rkg_method) + sep_suffix + "pkl")
            with open(rkgfile, 'rb') as f:
                rkg_ret = pickle.load(f)

            ret = {'object': [], 'SDR': {"ori": [], "sep": []}, 'SIR': {"ori": [], "sep": []}}
            for file_idx, (((utt_id_1, utt_path_1, offset_1, duration_1), (utt_id_2, utt_path_2, offset_2, duration_2)), _, _) in enumerate(makemix[0: 120]):
                print(file_idx)
                assert sep_ret['mix'][file_idx] == makemix[file_idx]
                perm_oracle = sep_ret['PERM']['sep'][file_idx]
                assert np.sum(perm_oracle) == 1
                rkg = rkg_ret['rkg'][2 * file_idx: 2 * file_idx + 2]
                target_idx = [-1, -1]
                if rkg[0] is True:
                    target_idx[0] = perm_oracle[0]
                else:
                    target_idx[0] = perm_oracle[1]
                if rkg[1] is True:
                    target_idx[1] = perm_oracle[1]
                else:
                    target_idx[1] = perm_oracle[0]
                assert len(target_idx) == 2
                ret['object'].append(makemix[file_idx])

                src_name_1 = os.path.basename(utt_path_1).split('.', 1)[0].rsplit('-', 1)[1]
                src_name_2 = os.path.basename(utt_path_2).split('.', 1)[0].rsplit('-', 1)[1]
                srcdata_path_1 = os.path.join(hp.path.script_prefix, utt_path_1)
                srcdata_path_2 = os.path.join(hp.path.script_prefix, utt_path_2)
                # src_name_1 = os.path.basename(utt_path_1).split('.', 1)[0][-4: ]
                # src_name_2 = os.path.basename(utt_path_2).split('.', 1)[0][-4: ]
                # srcdata_path_1 = os.path.join(hp.path.script_prefix, 'DATASET/Librispeech/concatenate_MN/test_dev_clean', 'speaker-' + src_name_1 + '.wav')
                # srcdata_path_2 = os.path.join(hp.path.script_prefix, 'DATASET/Librispeech/concatenate_MN/test_dev_clean', 'speaker-' + src_name_2 + '.wav')
                src_1, _ = sf.read(srcdata_path_1, start=offset_1, stop=offset_1 + duration_1)
                src_2, _ = sf.read(srcdata_path_2, start=offset_2, stop=offset_2 + duration_2)
                src_1 = src_1 / np.max(np.abs(src_1))
                src_2 = src_2 * np.std(src_1) / np.std(src_2)
                assert src_1.shape[0] == src_2.shape[0]
                src = np.stack((src_1, src_2), axis=1)
                src = src[0: src_len, :]

                mix_path = os.path.join(os.path.dirname(datafile), "{}-{}_{}-{}_mix.".format(src_name_1, utt_id_1, src_name_2, utt_id_2) + sep_suffix + "wav")
                mix, _ = sf.read(mix_path)
                if sep_method.startswith("MVAE"):
                    src = zero_pad(src.T, 4, hop_length=hop)
                    mix = zero_pad(mix.T, 4, hop_length=hop)
                else:
                    mix = mix.T
                    src = src.T
                sdr_ori, sir_ori, _, _ = mir_eval.separation.bss_eval_sources(src, np.stack((mix[0, :], mix[0, :]), axis=0))
                ret["SDR"]['ori'].extend(sdr_ori.tolist())
                ret["SIR"]['ori'].extend(sir_ori.tolist())
                
                if rkg == [True, True]:
                    ret["SDR"]['sep'].extend(sep_ret["SDR"]['sep'][2 * file_idx: 2 * file_idx + 2])
                    ret["SIR"]['sep'].extend(sep_ret["SIR"]['sep'][2 * file_idx: 2 * file_idx + 2])
                else:
                    sep_path = os.path.join(target_path, "{}-{}_{}-{}_sep.".format(src_name_1, utt_id_1, src_name_2, utt_id_2) + sep_suffix + "wav")
                    sep, _ = sf.read(sep_path)
                    sep = sep.T
                    sdr_1, sir_1, _, _ = mir_eval.separation.bss_eval_sources(src, np.stack((sep[target_idx[0], :], sep[target_idx[0], :]), axis=0))
                    sdr_2, sir_2, _, _ = mir_eval.separation.bss_eval_sources(src, np.stack((sep[target_idx[1], :], sep[target_idx[1], :]), axis=0))
                    # sdr, sir, _, _ = mir_eval.separation.bss_eval_sources(src, sep)
                    ret["SDR"]['sep'].extend([sdr_1[0], sdr_2[1]])
                    ret["SIR"]['sep'].extend([sir_1[0], sir_2[1]])

            assert 2 * file_idx + 2 == len(sep_ret["SDR"]['sep']) == len(rkg_ret['rkg'])
            if sub_state == 0:
                with open(os.path.join(target_path, 'extract_metrics-{}-{}.'.format(dataname, rkg_method) + suffix + sep_suffix + 'pkl'), 'wb') as f:
                    pickle.dump(ret, f)


def get_extracted_hist_data(datafile_list, rkg_method, dataname_list, out_path, sub_state=0):
    """ Get the number of corrrectly extracted utterances and the total number of trials according to sdr bins """
    """ Corresponding to Table 3 """
    if sub_state == 0:
        f_txt = open(out_path + ".txt", 'w')
    out = {}
    total_metrics = {} 
    total_metrics_oracle = {}
    bins = [-25, -5, 0, 5, 10, 20]
    # bins = [-30, 0, 5, 10, 15, 20, 25, 32]
    min_metric = 0.0
    max_metric = 0.0
    for datafile in datafile_list:
        # print(datafile, file=f_txt)
        for rkg in rkg_method:
            if rkg not in out:
                out[rkg] = [0 for _ in range(len(bins) - 1)]
                total_metrics[rkg] = []
                total_metrics_oracle[rkg] = []
            for dataname in dataname_list:
                sepfile_oracle = os.path.join(datafile, "metrics-{}.pkl".format(dataname, rkg))
                with open(sepfile_oracle, 'rb') as f:
                    sep_ret_oracle = pickle.load(f)
                sepfile = os.path.join(datafile, "extract_metrics-{}-{}.0.pkl".format(dataname, rkg))
                with open(sepfile, 'rb') as f:
                    sep_ret = pickle.load(f)
                data = [sep_ret['SDR']['sep'][i] - sep_ret['SDR']['ori'][i] for i in range(len(sep_ret['SDR']['ori']))]
                data_oracle = [sep_ret_oracle['SDR']['sep'][i] - sep_ret['SDR']['ori'][i] for i in range(len(sep_ret['SDR']['ori']))]
                # data = sep_ret['SDR']['sep']
                min_metric = min(min_metric, min(data))
                max_metric = max(max_metric, max(data))
                total_metrics[rkg].extend(data)
                total_metrics_oracle[rkg].extend(data_oracle)
                data_hist, _ = np.histogram(data, bins=bins)
                out[rkg] = out[rkg] + data_hist
    print("sdr bins: {}".format(bins), file=f_txt)
    for rkg in rkg_method:
        assert len(total_metrics[rkg]) == sum(out[rkg]) == len(total_metrics_oracle[rkg]) == 2160
        print("method {}: Extracted SDR hist: {}, Avg SDR: {}, Oracle SDR: {}".format(rkg, out[rkg], sum(total_metrics[rkg]) / len(total_metrics[rkg]), sum(total_metrics_oracle[rkg]) / len(total_metrics_oracle[rkg])), file=f_txt)
    if sub_state == 0:
        with open(out_path, 'wb') as f:
            pickle.dump(out, f)


def get_different_len_data(datafile_path, rkg_method, dataname, out_path, sep_suffix_list, sub_state=0):
    out = {}
    out['oracle'] = {'SDR': np.zeros(len(sep_suffix_list)), 'SIR': np.zeros(len(sep_suffix_list))}
    for sep_suffix_idx, sep_suffix in enumerate(sep_suffix_list):
        for datafile in datafile_path:
            metrics_path = os.path.join(datafile, "metrics-{}.".format(dataname) + sep_suffix + 'pkl')
            with open(metrics_path, 'rb') as f:
                sep_ret = pickle.load(f)
            
            for rkg_idx, rkg in enumerate(rkg_method):
                if rkg not in out:
                    out[rkg] = {'SDR': np.zeros(len(sep_suffix_list)), 'SIR': np.zeros(len(sep_suffix_list))}
                extract_metrics_path = os.path.join(datafile, "extract_metrics-{}-{}.0.".format(dataname, rkg) + sep_suffix + 'pkl')
                with open(extract_metrics_path, 'rb') as f:
                    extract_sep_ret = pickle.load(f)
                if rkg_idx == 0:
                    oracle_sdr = stat.mean([sep_ret['SDR']['sep'][i] - extract_sep_ret['SDR']['ori'][i] for i in range(len(extract_sep_ret['SDR']['ori']))])
                    oracle_sir = stat.mean([sep_ret['SIR']['sep'][i] - extract_sep_ret['SIR']['ori'][i] for i in range(len(extract_sep_ret['SIR']['ori']))])
                    out['oracle']['SDR'][sep_suffix_idx] += oracle_sdr
                    out['oracle']['SIR'][sep_suffix_idx] += oracle_sir
                aug_sdr = stat.mean([extract_sep_ret['SDR']['sep'][i] - extract_sep_ret['SDR']['ori'][i] for i in range(len(extract_sep_ret['SDR']['ori']))])
                aug_sir = stat.mean([extract_sep_ret['SIR']['sep'][i] - extract_sep_ret['SIR']['ori'][i] for i in range(len(extract_sep_ret['SIR']['ori']))])
                out[rkg]['SDR'][sep_suffix_idx] += aug_sdr
                out[rkg]['SIR'][sep_suffix_idx] += aug_sir
    out['oracle']['SDR'] = out['oracle']['SDR'] / len(datafile_path)
    out['oracle']['SIR'] = out['oracle']['SIR'] / len(datafile_path)
    for rkg in rkg_method:
        out[rkg]['SDR'] = out[rkg]['SDR'] / len(datafile_path)
        out[rkg]['SIR'] = out[rkg]['SIR'] / len(datafile_path)
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)


def get_rkg_ret(datafile_list, device, sep_method, suffix='0', spkr_encoder="ge2e_withaug", rkg_method="cossim",
                enroll_len=30, fs=16000, spkr_model=None, use_length_norm=True, sub_state=0, sep_suffix='',
                xvecmodel_path=None, ge2emodel_path=None, xvecmean_path=None, lda_path=None, plda_path=None,
                **kwargs):
    """ 
    Args:
        `datafile_list`: [str] path where makemix_xxx.pkl is saved
        `sep_method`: [str] name of separation method

    Out:
        `ret`: [Not returned but saved] 
               {"0.15-20": {"object": [file_idx-1-0, file_idx-1-1, file_idx-2-0, ...], "rkg": [rkg-for-file_idx-1-0, rkg-for-file_idx-1-1, ...]}}
    """
    if sep_suffix:
        src_len = int(int(sep_suffix) * fs)
        sep_suffix = str(sep_suffix) + 's.'
    else:
        src_len = int(int(30) * fs)
    ret = {'object': [], 'rkg': []}
    print(sep_method + '\n')

    print(spkr_encoder)
    if spkr_encoder.startswith('xvec'):
        print("speakermodel_path: {}".format(xvecmodel_path))
    elif spkr_encoder.startswith("ge2e"):
        print("speakermodel_path: {}".format(ge2emodel_path))
    print('\n')

    print(rkg_method)
    if rkg_method.startswith("plda"):
        print("xvecmean_path: {}".format(xvecmean_path))
        print("lda_path: {}".format(lda_path))
        print("plda_path: {}".format(plda_path))
        print("use_length_normalization: {}".format(use_length_norm))
    print('\n')
    
    print("enrollment length: {}".format(enroll_len))
    print(sep_suffix[: -1])

    # Prepare
    if spkr_encoder.startswith('ge2e'):
        fb_mat = torch.from_numpy(librosa.filters.mel(hp.data.sr, n_fft, n_mels=hp.model.nmels)).unsqueeze(0).cuda(device)
    elif spkr_encoder.startswith('xvec'):
        melsetting = {}
        melsetting['n_fft'] = n_fft
        melsetting['win_length'] = n_fft
        melsetting['hop_length'] = hop
        melsetting['n_mels'] = hp.model.feat_num
        transform = torchaudio.transforms.MFCC(sample_rate=hp.data.sr, n_mfcc=hp.model.feat_num, melkwargs=melsetting)
    # Load speaker model
    if spkr_encoder.startswith('ge2e'):
        spkr_model = speaker_encoder_ge2e()
        spkr_model_dict = spkr_model.state_dict()
        pretrained_dict = torch.load(os.path.join(hp.path.script_prefix, ge2emodel_path), map_location=torch.device('cpu'))
        pretrained_dict_rename = {}
        for k, v in pretrained_dict.items():
            try:
                param_name = k.split('.', 1)[1]
                pretrained_dict_rename[param_name] = v
            except IndexError:
                pass
    elif spkr_encoder.startswith('xvec'):
        spkr_model = speaker_encoder_xvec()
        spkr_model_dict = spkr_model.state_dict()
        pretrained_dict = torch.load(os.path.join(hp.path.script_prefix, xvecmodel_path), map_location=torch.device('cpu'))
        pretrained_dict_rename = pretrained_dict
    pretrained_dict_rename = {k: v for k, v in pretrained_dict_rename.items() if k in spkr_model_dict}
    spkr_model_dict.update(pretrained_dict_rename)
    spkr_model.load_state_dict(spkr_model_dict)
    spkr_model.cuda(device)
    spkr_model.eval()

    # recognize one by one
    for datafile in datafile_list:
        print(datafile)
        with open(datafile, 'rb') as f:
            makemix = pickle.load(f)
    
        dataname = os.path.basename(datafile)[0: -4]   # e.g. makemix_same_00
        sep_file = os.path.join(os.path.join(os.path.dirname(datafile), sep_method), 'metrics-{}.bak.'.format(dataname) + sep_suffix + 'pkl')
        with open(sep_file, 'rb') as f:
            sep_ret = pickle.load(f)

        ret = {"object": [], "rkg": []}
        met = {'object': [], 'SDR': {"ori": [], "sep": []}, 'SIR': {"ori": [], "sep": []}}
        target_path = os.path.join(os.path.dirname(datafile), sep_method)  # path where separated .wav is saved
        if sub_state == 0:
            # if os.path.exists(os.path.join(target_path, 'rkg_ret-{}-{}-{}.info.'.format(dataname, spkr_encoder, rkg_method) + suffix + '.txt')):
            #     os.system("mv {} {}".format(os.path.join(target_path, 'rkg_ret-{}-{}-{}.info.'.format(dataname, spkr_encoder, rkg_method) + suffix + '.txt'), os.path.join(target_path, 'rkg_ret-{}-{}-{}.info.'.format(dataname, spkr_encoder, rkg_method) + suffix + '.bak.txt')))
            # if os.path.exists(os.path.join(target_path, 'rkg_log-{}-{}-{}.'.format(dataname, spkr_encoder, rkg_method) + suffix + '.txt')):
            #     os.system("mv {} {}".format(os.path.join(target_path, 'rkg_log-{}-{}-{}.'.format(dataname, spkr_encoder, rkg_method) + suffix + '.txt'), os.path.join(target_path, 'rkg_log-{}-{}-{}.'.format(dataname, spkr_encoder, rkg_method) + suffix + '.bak.txt')))
            with open(os.path.join(target_path, 'rkg-{}-{}-{}.info.'.format(dataname, spkr_encoder, rkg_method) + suffix + '.' + sep_suffix + 'txt'), 'w') as f:
                print(sep_method + '\n', file=f)
                print(spkr_encoder, file=f)
                if spkr_encoder.startswith('xvec'):
                    print("speakermodel_path: {}".format(xvecmodel_path), file=f)
                elif spkr_encoder.startswith("ge2e"):
                    print("speakermodel_path: {}".format(ge2emodel_path), file=f)
                print('\n', file=f)
                print(rkg_method, file=f)
                if rkg_method.startswith("plda"):
                    print("xvecmean_path: {}".format(xvecmean_path), file=f)
                    print("lda_path: {}".format(lda_path), file=f)
                    print("plda_path: {}".format(plda_path), file=f)
                    print("use_length_normalization: {}".format(use_length_norm), file=f)
                print('\n', file=f)
                print("enrollment length: {}".format(enroll_len), file=f)

            f_txt = open(os.path.join(target_path, 'rkg-{}-{}-{}.log'.format(dataname, spkr_encoder, rkg_method) + suffix + '.' + sep_suffix + 'txt'), 'wt')

        for file_idx, (((utt_id_1, utt_path_1, offset_1, duration_1), (utt_id_2, utt_path_2, offset_2, duration_2)), _, _) in enumerate(makemix):
            print(file_idx)
            ret['object'].append(makemix[file_idx])
            assert sep_ret['mix'][file_idx] == makemix[file_idx]
            src_name_1 = os.path.basename(utt_path_1).split('.', 1)[0].rsplit('-', 1)[1]
            src_name_2 = os.path.basename(utt_path_2).split('.', 1)[0].rsplit('-', 1)[1]
            # src_name_1 = os.path.basename(utt_path_1)[0: -4]
            # src_name_2 = os.path.basename(utt_path_2)[0: -4]
            sep_path = os.path.join(target_path, "{}-{}_{}-{}_sep.".format(src_name_1, utt_id_1, src_name_2, utt_id_2) + sep_suffix + "wav")
            srcdata_path_1 = os.path.join(hp.path.script_prefix, utt_path_1)
            srcdata_path_2 = os.path.join(hp.path.script_prefix, utt_path_2)
            # srcdata_path_1 = os.path.join(hp.path.script_prefix, 'DATASET/Librispeech/concatenate/test_clean/' + src_name_1 + '.wav')
            # srcdata_path_2 = os.path.join(hp.path.script_prefix, 'DATASET/Librispeech/concatenate/test_clean/' + src_name_2 + '.wav')
            src_1, _ = sf.read(srcdata_path_1, start=offset_1, stop=offset_1 + duration_1)
            src_2, _ = sf.read(srcdata_path_2, start=offset_2, stop=offset_2 + duration_2)
            src_1 = src_1 / np.max(np.abs(src_1))
            src_2 = src_2 * np.std(src_1) / np.std(src_2)
            assert src_1.shape[0] == src_2.shape[0]
            src = np.stack((src_1, src_2), axis=1)
            src = src[0: src_len, :]

            mix_path = os.path.join(os.path.dirname(datafile), "{}-{}_{}-{}_mix.".format(src_name_1, utt_id_1, src_name_2, utt_id_2) + sep_suffix + "wav")
            mix, _ = sf.read(mix_path)
            if sep_method.startswith("MVAE"):
                src = zero_pad(src.T, 4, hop_length=hop)
                mix = zero_pad(mix.T, 4, hop_length=hop)
                src = src.T
                mix = mix.T

            enroll_1, _ = sf.read(srcdata_path_1, start=0, stop=int(enroll_len * fs))
            enroll_2, _ = sf.read(srcdata_path_2, start=0, stop=int(enroll_len * fs))
            enroll = np.stack((enroll_1, enroll_2), axis=1)
            sep, _ = sf.read(sep_path)
            sep_nm = copy.deepcopy(sep)
            src_nm = copy.deepcopy(src)
            # _, _, _, perm = mir_eval.separation.bss_eval_sources(src_nm.T, sep_nm.T)
            perm = sep_ret["PERM"]['sep'][file_idx]
            sdr_ori, sir_ori, _, _ = mir_eval.separation.bss_eval_sources(src_nm.T, np.stack((mix[:, 0], mix[:, 0]), axis=0))
            met["SDR"]['ori'].extend(sdr_ori.tolist())
            met["SIR"]['ori'].extend(sir_ori.tolist()) #.tolist()：把array转化成list

            sep = torch.from_numpy(sep.T)
            enroll = torch.from_numpy(enroll.T)
            with torch.no_grad():
                if spkr_encoder.startswith('xvec'):
                    sep = transform(sep.float())
                    sep = sep.float().cuda(device)
                    sep = (sep - sep.mean(dim=-1, keepdim=True)) / (sep.std(dim=-1, keepdim=True))
                    enroll = transform(enroll.float())
                    enroll = enroll.float().cuda(device)
                    enroll = (enroll - enroll.mean(dim=-1, keepdim=True)) / (enroll.std(dim=-1, keepdim=True))
                elif spkr_encoder.startswith('ge2e'):
                    sep = my_spectrogram(sep.float().cuda(device), n_fft, hop)
                    sep = spectrogram_normalize(sep)
                    sep = torch.matmul(fb_mat, sep)
                    sep = 10 * torch.log10(torch.clamp(sep, 1e-10))
                    enroll = my_spectrogram(enroll.float().cuda(device), n_fft, hop)
                    enroll = spectrogram_normalize(enroll)
                    enroll = torch.matmul(fb_mat, enroll)
                    enroll = 10 * torch.log10(torch.clamp(enroll, 1e-10))
                vec_sep = spkr_model.extract_embd(sep, use_slide=True)
                vec_enroll = spkr_model.extract_embd(enroll, use_slide=True)

                if rkg_method == "cossim":
                    sim_enroll_1 = F.cosine_similarity(vec_sep, vec_enroll[0, None, :], dim=1)
                    sim_enroll_2 = F.cosine_similarity(vec_sep, vec_enroll[1, None, :], dim=1)
                    target_idx_enroll_1 = sim_enroll_1.argmax()
                    target_idx_enroll_2 = sim_enroll_2.argmax()

                elif rkg_method.startswith("plda"):
                    vec_sep = vec_sep.cpu().numpy()
                    vec_enroll = vec_enroll.cpu().numpy()
                    embd_mean = kwargs["embedding_mean"]
                    lda_machine = kwargs["lda_machine"]
                    plda_base = kwargs["plda_base"]
                    vec_sep = lda_machine.forward(vec_sep - embd_mean)
                    vec_enroll = lda_machine.forward(vec_enroll - embd_mean)
                    if use_length_norm:
                        vec_sep = vec_sep / np.linalg.norm(vec_sep, axis=1, keepdims=True)
                        vec_enroll = vec_enroll / np.linalg.norm(vec_sep, axis=1, keepdims=True)
                    vec_sep = vec_sep.astype(np.float64)
                    vec_enroll = vec_enroll.astype(np.float64)
                    plda_machine_1 = bob.learn.em.PLDAMachine(plda_base)
                    plda_machine_2 = bob.learn.em.PLDAMachine(plda_base)
                    plda_trainer = bob.learn.em.PLDATrainer()
                    plda_trainer.enroll(plda_machine_1, vec_enroll[0, None, :])
                    plda_trainer.enroll(plda_machine_2, vec_enroll[1, None, :])
                    loglike_enroll_1 = np.stack([plda_machine_1.compute_log_likelihood(vec_sep[0, :]), plda_machine_1.compute_log_likelihood(vec_sep[1, :])])
                    loglike_enroll_2 = np.stack([plda_machine_2.compute_log_likelihood(vec_sep[0, :]), plda_machine_2.compute_log_likelihood(vec_sep[1, :])])
                    # loglike_enroll_1 = np.stack([
                    #     plda_machine.compute_log_likelihood(np.stack((vec_enroll[0, :], vec_sep[0, :]))), 
                    #     plda_machine.compute_log_likelihood(np.stack((vec_enroll[0, :], vec_sep[1, :])))])
                    # loglike_enroll_2 = np.stack([
                    #     plda_machine.compute_log_likelihood(np.stack((vec_enroll[1, :], vec_sep[0, :]))), 
                    #     plda_machine.compute_log_likelihood(np.stack((vec_enroll[1, :], vec_sep[1, :])))])
                    target_idx_enroll_1 = loglike_enroll_1.argmax()
                    target_idx_enroll_2 = loglike_enroll_2.argmax()

                ret['rkg'].append(True if target_idx_enroll_1 == perm[0] else False)
                ret['rkg'].append(True if target_idx_enroll_2 == perm[1] else False)

                sdr_1, sir_1, _, _ = mir_eval.separation.bss_eval_sources(src_nm.T, np.stack((sep_nm[:, target_idx_enroll_1], sep_nm[:, target_idx_enroll_1]), axis=0))
                sdr_2, sir_2, _, _ = mir_eval.separation.bss_eval_sources(src_nm.T, np.stack((sep_nm[:, target_idx_enroll_2], sep_nm[:, target_idx_enroll_2]), axis=0))
                met["SDR"]['sep'].extend([sdr_1[0], sdr_2[1]])
                met["SIR"]['sep'].extend([sir_1[0], sir_2[1]])
                if sub_state == 0:
                    print("{}-{}_{}-{}_{}_sep.wav: {}\t{}\t{:.2f}\t{:.2f}".format(src_name_1, utt_id_1, src_name_2, utt_id_2, sep_method,
                                                                  target_idx_enroll_1 == perm[0],
                                                                  target_idx_enroll_2 == perm[1], sdr_1[0], sdr_2[1]), file=f_txt)
        if sub_state == 0:
            # if os.path.exists(os.path.join(target_path, 'rkg_ret-{}-{}-{}.'.format(dataname, spkr_encoder, rkg_method) + suffix + '.pkl')):
            #     os.system("mv {} {}".format(os.path.join(target_path, 'rkg_ret-{}-{}-{}.'.format(dataname, spkr_encoder, rkg_method) + suffix + '.pkl'), os.path.join(target_path, 'rkg_ret-{}-{}-{}.' + suffix + '.bak.pkl').format(dataname, spkr_encoder, rkg_method)))
            with open(os.path.join(target_path, 'rkg_ret-{}-{}-{}.'.format(dataname, spkr_encoder, rkg_method) + suffix + '.' + sep_suffix + 'pkl'), 'wb') as f:
                pickle.dump(ret, f)
            with open(os.path.join(target_path, 'extract_metrics-{}-{}-{}.'.format(dataname, spkr_encoder, rkg_method) + suffix + '.' + sep_suffix + 'pkl'), 'wb') as f:
                pickle.dump(met, f)


def concate_rkg_ret(datafile_list, concate_ret_path):
    """ print recognition results of all settings into a single file """
    """ 
    Args:
        datafile_list: list of which each element is a path for recognition results
        concate_ret_path: txt file where results are printed
    """
    if os.path.exists(concate_ret_path):
        os.system("cp {} {}".format(concate_ret_path, concate_ret_path[0: -4] + '_bak.pkl'))
        with open(concate_ret_path, 'rb') as f:
            concate_ret = pickle.load(f)
    else:
        concate_ret = {}
    for datafile in datafile_list:
        with open(datafile, 'rb') as f:
            ret = pickle.load(f)
        env = os.path.basename(os.path.dirname(os.path.dirname(datafile)))
        _, dataname, spkr_encoder, rkg_meth = os.path.basename(datafile).split('-')
        rkg_meth = rkg_meth.rsplit('.')[0]
        meth = spkr_encoder + "-" + rkg_meth
        if env not in concate_ret:
            concate_ret[env] = {}
        if meth not in concate_ret[env]:
            concate_ret[env][meth] = {}
        if dataname not in concate_ret[env][meth]:
            concate_ret[env][meth][dataname] = [sum(ret["rkg"]), len(ret["rkg"])]
    with open(concate_ret_path, 'wb') as f:
        pickle.dump(concate_ret, f)

    concate_ret_all = {}
    f_txt = open(concate_ret_path[0: -4] + '.txt', 'wt')
    for env in concate_ret.keys():
        if env not in concate_ret_all:
            concate_ret_all[env] = {}
        print("Environment: {}".format(env), file=f_txt)
        for meth in concate_ret[env].keys():
            total_right = 0
            total_num = 0
            for dataname in concate_ret[env][meth].keys():
                if len(concate_ret[env][meth].keys()) > 1:
                    print("\t{}-{}: {}/{}".format(meth, dataname, concate_ret[env][meth][dataname][0], concate_ret[env][meth][dataname][1]), file=f_txt)
                total_num += concate_ret[env][meth][dataname][1]
                total_right += concate_ret[env][meth][dataname][0]
            print("Total recognition result for {}: {}/{}".format(meth, total_right, total_num), file=f_txt)
            concate_ret_all[env][meth] = (total_right, total_num, total_right / total_num)
        print('\n', file=f_txt)
    f_txt.close()
    with open(os.path.join(os.path.dirname(concate_ret_path), 'All-' + os.path.basename(concate_ret_path)), 'wb') as f:
        pickle.dump(concate_ret_all, f)
    

def concate_sep_ret(datafile_list, concate_ret_path):
    """ concatenate separation results into a single file """
    """ 
    Args:
        `datafile_list`: a list of which each element is a path to separation results
        `concate_ret_path`: list  where the final concatenated results will be saved
    """
    if os.path.exists(concate_ret_path):
        concate_ret = {}
        # os.system("cp {} {}".format(concate_ret_path, concate_ret_path[0: -4] + '_bak.pkl'))
        # with open(concate_ret_path, 'rb') as f:
        #     concate_ret = pickle.load(f)
    else:
        concate_ret = {}
    for datafile in datafile_list:
        with open(datafile, 'rb') as f:
            ret = pickle.load(f)
        env = os.path.basename(os.path.dirname(os.path.dirname(datafile)))
        dataname = os.path.basename(datafile).split('-')[1].split('.')[0]
        if env not in concate_ret:
            concate_ret[env] = {}
        if dataname not in concate_ret[env]:
            concate_ret[env][dataname] = {}
            for m in ['SDR', 'SIR', 'SAR']:
                concate_ret[env][dataname][m] = [stat.mean(ret[m]['sep']), stat.mean([ret[m]['sep'][i] - ret[m]['ori'][i] for i in range(0, len(ret[m]['sep']), 3)]), len(ret[m]['sep'])]
    with open(concate_ret_path, 'wb') as f:
        pickle.dump(concate_ret, f)

    concate_ret_all = {}
    sdr_all_all = 0.0
    sir_all_all = 0.0
    total_num_all = 0.0
    f_txt = open(concate_ret_path[0: -4] + '.txt', 'wt')
    for env in concate_ret.keys():
        print("Environment: {}".format(env), file=f_txt)
        sdr_all = 0.0
        sir_all = 0.0
        sar_all = 0.0
        total_num = 0
        for dataname in concate_ret[env].keys():
            if len(concate_ret[env].keys()) > 1:
                print("\t{}: SDR improvements: {:.2f}, SIR improvements: {:.2f}, SAR improvements: {:.2f}".format(dataname, concate_ret[env][dataname]["SDR"][0],
                                                                                                                  concate_ret[env][dataname]["SIR"][0],
                                                                                                                  concate_ret[env][dataname]["SAR"][0], file=f_txt))
            total_num += concate_ret[env][dataname]["SDR"][-1]
            sdr_all += (concate_ret[env][dataname]["SDR"][0] * concate_ret[env][dataname]["SDR"][-1])
            sir_all += (concate_ret[env][dataname]["SIR"][0] * concate_ret[env][dataname]["SDR"][-1])
            sar_all += (concate_ret[env][dataname]["SAR"][0] * concate_ret[env][dataname]["SDR"][-1])
        print("Total recognition result for {}: SDR improvements: {:.2f}, SIR improvements: {:.2f}, SAR improvements: {:.2f}".format(env,
                                                                                                                                     sdr_all / total_num,
                                                                                                                                     sir_all / total_num,
                                                                                                                                     sar_all / total_num), file=f_txt)
        print('\n', file=f_txt)
        if env not in ["0.160-30_90_150", "0.360-30_90_150", "0.610-30_90_150"]:
            total_num_all += total_num
            sdr_all_all += sdr_all
            sir_all_all += sir_all
    sdr_all_all /= total_num_all
    sir_all_all /= total_num_all
    f_txt.close()


def get_sep_basedon_rkg(datafile_list, dataname_list, rkg_method, out_path):
    """ concatenate separation results into a single file """
    """ 
    Args:
        `datafile_list`: a list of which each element is a path to separation results
        `out_path`: list  where the final concatenated results will be saved
    """
    out_sep = {}  # {'rkg_method': {'ILRMA': {'0.15-20': {"SIR": [seped, improved, num], "SDR": [seped, improved, num]}, '0.15-30': {}, ...}, ...}, 'MVAE_onehot': {}}, ...}
    out_rkg = {}  # {'rkg_method': {'ILRMA': {'0.15-20': [correct, all], '0.15-30': [correct, all]}}}, 'MVAE_onehot': {}}
    os.makedirs(out_path, exist_ok=True)
    for rkg in rkg_method:
        # create rkg_method keys
        if rkg not in out_sep:
            out_sep[rkg] = {}
        if rkg not in out_rkg:
            out_rkg[rkg] = {}
        # iterate through env and sep_meth
        for datafile in datafile_list:
            # create 
            env = os.path.basename(os.path.dirname(datafile))
            sep_method = os.path.basename(datafile)
            if sep_method not in out_sep[rkg]:
                out_sep[rkg][sep_method] = {}
            if env not in out_sep[rkg][sep_method]:
                out_sep[rkg][sep_method][env] = {"SIR": [], "SDR": []}
            if sep_method not in out_rkg[rkg]:
                out_rkg[rkg][sep_method] = {}
            if env not in out_rkg[rkg][sep_method]:
                out_rkg[rkg][sep_method][env] = [0, 0]
            # iterates through datasets
            sdr_all = 0.0
            sir_all = 0.0
            sdr_imp_all = 0.0
            sir_imp_all = 0.0
            sdr_all_correct = 0.0
            sir_all_correct = 0.0
            sdr_imp_all_correct = 0.0
            sir_imp_all_correct = 0.0
            correct_num = 0
            total_num = 0
            for dataname in dataname_list:
                sepfile = os.path.join(datafile, "metrics-{}.pkl".format(dataname))
                rkgfile = os.path.join(datafile, "rkg_ret-{}-{}.pkl".format(dataname, rkg))
                with open(sepfile, 'rb') as f:
                    sep_ret = pickle.load(f)  # {'mix': [], 'SIR': {'sep': [], 'ori': []}}
                with open(rkgfile, 'rb') as f:
                    rkg_ret = pickle.load(f)
                assert len(rkg_ret["rkg"]) == len(sep_ret['SIR']['sep']) == len(rkg_ret["object"])
                # locate the index of the correct extraction
                correct_idx = [i for i, flag in enumerate(rkg_ret['rkg']) if flag]
                # cumulate the sir and sdr results for both correct and all situation
                """ all """
                sdr_all += sum(sep_ret['SDR']['sep'])
                sir_all += sum(sep_ret['SIR']['sep'])
                sdr_imp_all += sum([sep_ret['SDR']['sep'][m] - sep_ret['SDR']['ori'][m] for m in range(len(sep_ret['SDR']['sep']))])
                sir_imp_all += sum([sep_ret['SIR']['sep'][m] - sep_ret['SIR']['ori'][m] for m in range(len(sep_ret['SIR']['sep']))])
                """ only the correct """
                sdr_all_correct += sum([sep_ret['SDR']['sep'][m] for m in correct_idx])
                sir_all_correct += sum([sep_ret['SIR']['sep'][m] for m in correct_idx])
                sdr_imp_all_correct += sum([sep_ret['SDR']['sep'][m] - sep_ret['SDR']['ori'][m] for m in correct_idx])
                sir_imp_all_correct += sum([sep_ret['SIR']['sep'][m] - sep_ret['SIR']['ori'][m] for m in correct_idx])
                """ the correct and total number """
                correct_num += len(correct_idx)
                total_num += len(rkg_ret['rkg'])
            out_sep[rkg][sep_method][env]["SIR"] = [
                [sir_all_correct / correct_num, sir_imp_all_correct / correct_num, correct_num],
                [sir_all / total_num, sir_imp_all / total_num, total_num]
            ]
            out_sep[rkg][sep_method][env]["SDR"] = [
                [sdr_all_correct / correct_num, sdr_imp_all_correct / correct_num, correct_num],
                [sdr_all / total_num, sdr_imp_all / total_num, total_num]
            ]
            out_rkg[rkg][sep_method][env] = [correct_num, total_num]
    # with open(os.path.join(out_path, 'rkg_ret.pkl'), 'wb') as f:
    #     pickle.dump(out_rkg, f)
    with open(os.path.join(out_path, 'sep_ret_MVAE_onehot_ilrmainit.pkl'), 'wb') as f:
        pickle.dump(out_sep, f)


def get_hist_data(datafile_list, rkg_method, dataname_list, out_path, sub_state=0):
    """ Get the number of corrrectly extracted utterances and the total number of trials according to sdr bins """
    """ Corresponding to Table 3 """
    # if os.path.exists(out_path + '.txt'):
    #     os.system("mv {} {}".format(out_path + ".txt", out_path + ".bak.txt"))
    if sub_state == 0:
        f_txt = open(out_path + ".txt", 'w')
    out = {}
    total_metrics = {}
    # bins = [-12, 0, 4, 8, 12, 16, 20, 24, 32]
    # bins = [-15, -5, 5, 15, 25, 35]
    # bins = [-12, 0, 5, 10, 15, 20, 25, 35]
    bins = [-15, -5, 0, 5, 10, 20]
    min_metric = 0.0
    max_metric = 0.0
    for datafile in datafile_list:
        # print(datafile, file=f_txt)
        for dataname in dataname_list:
            sepfile = os.path.join(datafile, "metrics-{}.pkl".format(dataname))
            with open(sepfile, 'rb') as f:
                sep_ret = pickle.load(f)
            sep_ret['mix'] = [i for i in sep_ret['mix'] for _ in range(2)]
            min_metric = min(min_metric, min(sep_ret['SDR']['sep']))
            max_metric = max(max_metric, max(sep_ret['SDR']['sep']))
            for rkg in rkg_method:
                data_correct = []
                if rkg not in out:
                    out[rkg] = [np.zeros(len(bins) - 1) for _ in range(2)]
                    total_metrics[rkg] = []
                rkgfile = os.path.join(datafile, "rkg_ret-{}-{}.0.pkl".format(dataname, rkg))
                with open(rkgfile, 'rb') as f:
                    rkg_ret = pickle.load(f)
                rkg_ret["object"] = [x.split('-')[0] for x in rkg_ret["object"]]
                assert len(rkg_ret["rkg"]) == len(sep_ret['SIR']['sep']) == len(rkg_ret["object"])
                data_all = copy.deepcopy(sep_ret["SDR"]['sep'])
                total_metrics[rkg].extend([sep_ret['SDR']['sep'][i] - sep_ret['SDR']['ori'][i] for i in range(len(sep_ret['SDR']['ori']))])
                for i in range(len(rkg_ret['object'])):
                    if rkg_ret["rkg"][i]:
                        data_correct.append(sep_ret["SDR"]['sep'][i])
                # out['rkg'][0] = stat.mean(data_correct)
                data_all_hist, _ = np.histogram(data_all, bins=bins)
                data_correct_hist, _ = np.histogram(data_correct, bins=bins)
                out[rkg][0] = out[rkg][0] + data_correct_hist
                out[rkg][1] = out[rkg][1] + data_all_hist
    print("sdr bins: {}".format(bins), file=f_txt)
    for rkg in rkg_method:
        assert len(total_metrics[rkg]) == sum(out[rkg][1]) == 2160
        ratio = out[rkg][0] / out[rkg][1]
        print("method {}: Correct: {}, All: {}, Ratio: {}, AvgOracleSDR: {}".format(rkg, out[rkg][0], out[rkg][1], ratio, sum(total_metrics[rkg]) / len(total_metrics[rkg])), file=f_txt)
        out[rkg].append(ratio)
    if sub_state == 0:
        # if os.path.exists(out_path):
        #     os.system("mv {} {}".format(out_path, out_path[0: -4] + ".bak.pkl"))
        with open(out_path, 'wb') as f:
            pickle.dump(out, f)


def get_extracted_metric_data(datafile_list, rkg_method, dataname_list, out_path, sub_state=0, suffix='', sep_suffix=''):
    """ Get SDR and SIRs of the successfully extracted utterances """
    """ Correspond to the "extracted speech quality" colunmn in Table 4 """
    if sep_suffix:
        sep_suffix = str(sep_suffix) + 's.'
    f_txt = open(out_path[0: -4] + suffix + ".txt", 'w')
    for datafile in datafile_list:
        print(datafile, file=f_txt)
        for dataname in dataname_list:
            sepfile_oracle = os.path.join(datafile, "metrics-{}.pkl".format(dataname))
            with open(sepfile_oracle, 'rb') as f:
                sep_ret_oracle = pickle.load(f)
            for rkg in rkg_method:
                sepfile = os.path.join(datafile, "extract_metrics-{}-{}.0.".format(dataname, rkg) + sep_suffix + "pkl")
                with open(sepfile, 'rb') as f:
                    sep_ret = pickle.load(f)
                sdr = [sep_ret['SDR']['sep'][i] - sep_ret['SDR']['ori'][i] for i in range(len(sep_ret['SDR']['sep']))]
                sir = [sep_ret['SIR']['sep'][i] - sep_ret['SIR']['ori'][i] for i in range(len(sep_ret['SIR']['sep']))]
                print('rkg_method: {} --- SDRi: {:.2f}, SIRi: {:.2f}'.format(rkg, stat.mean(sdr), stat.mean(sir)), file=f_txt)
                # sir = []
                # sdr = []
                # rkgfile = os.path.join(datafile, "rkg_ret-{}-{}.".format(dataname, rkg) + suffix + ".pkl")
                # with open(rkgfile, 'rb') as f:
                #     rkg_ret = pickle.load(f)
                # rkg_ret["object"] = [x.split('-')[0] for x in rkg_ret["object"]]
                # assert len(rkg_ret["rkg"]) == len(sep_ret['SIR']['sep']) == len(rkg_ret["object"])
                # for i in range(len(rkg_ret['object'])):
                #     if rkg_ret["rkg"][i]:
                #         sir.append(sep_ret['SIR']['sep'][i] - sep_ret['SIR']['ori'][i])  # improved
                #         sdr.append(sep_ret['SDR']['sep'][i] - sep_ret['SDR']['ori'][i])
                # print('rkg_method: {} --- SDR: {:.2f}, SIR: {:.2f}'.format(rkg, stat.mean(sdr), stat.mean(sir)), file=f_txt)
            sdr = [sep_ret_oracle['SDR']['sep'][i] - sep_ret['SDR']['ori'][i] for i in range(len(sep_ret_oracle['SDR']['sep']))]
            sir = [sep_ret_oracle['SIR']['sep'][i] - sep_ret['SIR']['ori'][i] for i in range(len(sep_ret_oracle['SIR']['sep']))]
            print('rkg_method: oracle --- SDRi: {:.2f}, SIRi: {:.2f}'.format(stat.mean(sdr), stat.mean(sir)), file=f_txt)
            print("\n", file=f_txt)


def sdr2sir(datafile_list, dataname_list, out_path):
    bins = [-8, -4, 0, 4, 8, 12, 16, 20, 24]
    sdr2sir = [[] for _ in range(len(bins) - 1)]
    for datafile in datafile_list:
        for dataname in dataname_list:
            sepfile = os.path.join(datafile, "metrics-{}.pkl".format(dataname))
            with open(sepfile, 'rb') as f:
                sep_ret = pickle.load(f)
            for idx, data in enumerate(sep_ret["SDR"]["sep"]):
                for i in range(len(bins) - 1):
                    if (data >= bins[i]) and (data < bins[i + 1]):
                        if i == 0 and sep_ret["SIR"]['sep'][idx] > 10:
                            import ipdb; ipdb.set_trace()
                        sdr2sir[i].append(sep_ret["SIR"]['sep'][idx])
    with open(out_path, 'wb') as f:
        pickle.dump(sdr2sir, f)
    f = open(out_path + '.txt', 'w')
    for i, sir_list in enumerate(sdr2sir):
        min_num = min(sir_list)
        max_num = max(sir_list)
        print("bins {}-{}: {:.2f}--{:.2f}".format(bins[i], bins[i + 1], min_num, max_num), file=f)
    f.close()


def get_all_rkg_data(datafile_list, rkg_method, dataname_list, out_path, suffix=''):
    if os.path.exists(out_path):
        os.system("mv {} {}".format(out_path, out_path[0:-4] + '.bak.txt'))
    f_txt = open(out_path, 'w')
    for datafile in datafile_list:
        out = {}
        print(datafile, file=f_txt)
        for dataname in dataname_list:
            for rkg in rkg_method:
                rkgfile = os.path.join(datafile, "rkg_ret-{}-{}.".format(dataname, rkg) + suffix + ".pkl")
                if rkg not in out:
                    out[rkg] = [0, 0]
                with open(rkgfile, 'rb') as f:
                    rkg_ret = pickle.load(f)
                out[rkg][0] = out[rkg][0] + sum(rkg_ret['rkg'])
                out[rkg][1] = out[rkg][1] + len(rkg_ret['rkg'])
        for rkg in rkg_method:
            print("{}: {} {} {:.2f}".format(rkg, out[rkg][0], out[rkg][1], out[rkg][0] * 100 / out[rkg][1]), file=f_txt)
        print("\n", file=f_txt)
        

def rkg2sdr(datafile_list, out_path):
    t60 = ["0.160", "0.360", "0.610"]
    sdr = ['-5', '0', '5']
    out_part = {"-5": [[0, 0] for _ in range(3)], "0": [[0, 0] for _ in range(3)], "5": [[0, 0] for _ in range(3)]}
    out_all = {"-5": [0, 0], "0": [0, 0], "5": [0, 0]}
    f_txt = open(out_path, 'wt')
    for file in datafile_list:
        print(file, file=f_txt)
        with open(file, 'rb') as f:
            ret = pickle.load(f)
        out = {"-5": [], "0": [], "5": []}
        for i, key in enumerate(sdr):
            out[key] = [sum(ret['rkg'][i::3]), len(range(i, len(ret['rkg']), 3))]
        idx = t60.index(os.path.basename(os.path.dirname(os.path.dirname(file))).split('-')[0])
        for key in sdr:
            out_part[key][idx][0] += out[key][0]
            out_part[key][idx][1] += out[key][1]
            out_all[key][0] += out[key][0]
            out_all[key][1] += out[key][1]
            print("{}: {}/{}".format(key, out[key][0], out[key][1]), file=f_txt)
        print("\n", file=f_txt)
    
    with open(out_path + '.part.txt', 'w') as f:
        for i, t in enumerate(t60):
            for key in sdr:
                print("{}-{}: {}/{}".format(t, key, out_part[key][i][0], out_part[key][idx][1]), file=f)
            print("\n", file=f)
    with open(out_path + '.all.txt', 'w') as f:
        for key in sdr:
            print("{}: {}/{}".format(key, out_all[key][0], out_all[key][1]), file=f)


def forgithubdemo(datafile_list):
    ret = []
    for file in datafile_list:
        with open(file, 'rb') as f:
            metrics_dict = pickle.load(f)
        ret.append(metrics_dict['SIR']['sep'])
    vad_list = list(filter(lambda x: x % 16 == 0 or x % 16 == 1, range(len(ret[0]))))
    diff = [ret[1][i] - ret[0][i] for i in vad_list]
    p = diff.index(max(diff))
    dummy = 1
        

def main(state=0, sub_state=0, script_prefix=None):
    "Interspeech论文结果主要用到state=0,3,4"
    if state == 0:
        """ Conduct recognition """
        sep_suffix_list = ['']
        # sep_suffix_list = [10, 15, 20, 25]
        # for sep_suffix in sep_suffix_list:
        #     # t60 = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
        #     t60 = [0.15, 0.25, 0.35]  #, 0.45, 0.55, 0.65
        #     # t60 = [0.45, 0.55, 0.65]  #, 
        #     angle_interval = [20, 30, 40, 70, 90, 110]
        #     # t60 = [0.15, 0.25, 0.35, 0.45]
        #     # angle_interval = [10, 50, 60, 80, 130]
        #     datafile_list = [
        #         os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu_remix/t60_angle_interval_study_withsir/{}-{}/makemix_same_00.pkl".format(x[0], x[1]))
        #         for x in product(t60, angle_interval)
        #     ]
        #     print("\ndon't forget to adjust makemix！！！！！！！！！\n")

        for sep_suffix in sep_suffix_list:
            t60 = [0.16, 0.36, 0.61]
            t60 = ["{:.3f}".format(x) for x in t60]
            datafile_list = [
                os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study_withsir/{}/makemix_same_00.pkl".format(x))
                for x in t60
            ]
            print("\ndon't forget to adjust makemix！！！！！！！！！\n")

            # t60_list = [0.16, 0.36, 0.61]
            # t60_list = ['{:.3f}'.format(x) for x in t60_list] #, [75, 120, 165], [15, 45, 75], [30, 45, 120], [60, 90, 105], [30, 90, 150], [30, 45, 60]
            # angle_list = [[90, 30, 150]]
            # angle_list = [[str(x) for x in a_list] for a_list in angle_list]
            # angle_list = ['_'.join(a_list) for a_list in angle_list]
            # datafile_list = [
            #     os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study_test/{}-{}/makemix_same_00.pkl".format(x[0], x[1]))
            #     for x in product(t60_list, angle_list)
            # ]

            """ ================Need check===================== """
            enroll_len = 30
            device = torch.device(hp.device)
            suffix = '2'
            sep_method = 'MVAE_onehot'
            spkr_encoder = "xvec_sepaug"
            rkg_method = ["cossim", "plda_withoutaug_onlyln", "plda_withaug_onlyln"][2]
            """ ================================================ """

            use_length_norm = True if rkg_method.endswith("onlyln") else False
            if spkr_encoder.endswith("sepaug"):
                xvecmodel_path = hp.path.xvecmodel_path[0]
                ge2emodel_path = None  # hp.path.ge2emodel_path[0]
            elif spkr_encoder.endswith("diraug"):
                xvecmodel_path = hp.path.xvecmodel_path[1]
                ge2emodel_path = None  # hp.path.ge2emodel_path
            elif spkr_encoder.endswith("withoutaug"):
                xvecmodel_path = hp.path.xvecmodel_path[2]
                ge2emodel_path = None  # hp.path.ge2emodel_path

            # for PLDA backend only
            plda_path = os.path.join(script_prefix, os.path.dirname(os.path.dirname(xvecmodel_path)), 'PLDA')

            """ ================Need check===================== """
            if rkg_method == "plda_withaug_onlyln":
                xvecmean_path = os.path.join(plda_path, "xvec_mean.train.augdata.pkl")
            elif rkg_method == "plda_withoutaug_onlyln":
                xvecmean_path = os.path.join(plda_path, "xvec_mean.train.oridata.pkl")
            lda_path = os.path.join(plda_path, "lda_machine.withdimreduction.0.hdf5")
            plda_path = os.path.join(plda_path, "plda_base.0.hdf5")
            """ ================================================ """

            with open(xvecmean_path, 'rb') as f:
                embedding_mean = pickle.load(f)
            machine_file = bob.io.base.HDF5File(lda_path)
            lda_machine = bob.learn.linear.Machine(machine_file)
            del machine_file
            whitening_matrix = None
            plda_hdf5 = bob.io.base.HDF5File(plda_path)
            plda_base = bob.learn.em.PLDABase(plda_hdf5)
            get_rkg_ret(datafile_list, device, sep_method, suffix=suffix, spkr_encoder=spkr_encoder, rkg_method=rkg_method, enroll_len=enroll_len, fs=16000, sub_state=sub_state, sep_suffix=sep_suffix,
                        xvecmodel_path=xvecmodel_path, ge2emodel_path=ge2emodel_path, use_length_norm=use_length_norm, xvecmean_path=xvecmean_path, lda_path=lda_path, plda_path=plda_path,
                        embedding_mean=embedding_mean, lda_machine=lda_machine, whitening_matrix=whitening_matrix, plda_base=plda_base)
    
    elif state == 1:
        """ Concatenate recognition result """
        method = [
            'rkg_ret-makemix_same_00-xvec_withaug-plda_withaug_onlyln',
            # 'rkg_ret-makemix_same_00-xvec_withoutaug-plda_withaug_onlyln',
            # 'rkg_ret-makemix_same_00-xvec_withoutaug-plda_withoutaug_onlyln',
            # 'rkg_ret-makemix_same_00-ge2e_withoutaug-cossim',
            # 'rkg_ret-makemix_same_00-ge2e_withaug-cossim',
        ]
        sep_method = 'MVAE_onehot_ilrmainit'
        suffix = '0'
        # t60 = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
        # angle_interval = [20, 30, 40, 70, 90, 110]
        # t60 = [0.15, 0.25, 0.35, 0.45]
        # angle_interval = [10, 50, 60, 80, 130]
        # datafile_list = [
        #     os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/extra/{}-{}/{}/{}.pkl".format(x[0], x[1], sep_method, x[2]))
        #     for x in product(t60, angle_interval, method)
        # ]
        # concate_ret_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/extra/rkg_ret--{}.pkl".format(sep_method))
        t60 = [0.16, 0.36, 0.61]
        t60 = ["{:.3f}".format(x) for x in t60]
        datafile_list = [
            os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/{}/{}/{}.".format(x[0], sep_method, x[1]) + suffix + ".pkl")
            for x in product(t60, method)
        ]
        concate_ret_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/rkg_ret--{}--xvec".format(sep_method) + suffix + ".pkl")
        # t60_list = [0.16, 0.36, 0.61]
        # t60_list = ['{:.3f}'.format(x) for x in t60_list]
        # # angle_list = [[30, 75, 120, 150], [30, 90, 120, 165], [15, 45, 75, 150], [30, 45, 105, 150], [30, 45, 105, 120]]
        # angle_list = [[30, 105, 150], [30, 90, 150], [75, 120, 165], [15, 45, 75], [30, 45, 120], [60, 90, 105], [30, 45, 60]]
        # angle_list = [[str(x) for x in a_list] for a_list in angle_list]
        # angle_list = ['_'.join(a_list) for a_list in angle_list]
        # datafile_list = [
        #     os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study-3_src-withsir-rirsmall-2/{}-{}/{}/{}.pkl".format(x[0], x[1], sep_method, x[2]))
        #     for x in product(t60_list, angle_list, method)
        # ]
        # concate_ret_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study-3_src-withsir-rirsmall/rkg_ret--{}--xvec.pkl".format(sep_method))
        concate_rkg_ret(datafile_list, concate_ret_path)

    elif state == 2:
        """ Concatenate separation result """
        sep_method = 'ilrma'
        t60 = [0.15, 0.25, 0.35, 0.45]
        angle_interval = [10, 50, 60, 80, 130]
        datafile_list = [
            os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/extra/{}-{}/{}/metrics-makemix_same_00.pkl".format(x[0], x[1], sep_method))
            for x in product(t60, angle_interval)
        ]
        concate_ret_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/extra/sep_ret--{}.pkl".format(sep_method))
        # t60 = [0.16, 0.36, 0.61]
        # t60 = ["{:.3f}".format(x) for x in t60]
        # datafile_list = [
        #     os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study/{}/{}/metrics-makemix_same_00.pkl".format(x, sep_method))
        #     for x in t60
        # ]
        # concate_ret_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study/sep_ret--{}.pkl".format(sep_method))
        # t60_list = [[0.16, 0.36, 0.61][0]]
        # t60_list = ['{:.3f}'.format(x) for x in t60_list]
        # angle_list = [[30, 105, 150], [30, 90, 150], [75, 120, 165], [15, 45, 75], [30, 45, 120], [60, 90, 105], [30, 45, 60]]
        # angle_list = [[str(x) for x in a_list] for a_list in angle_list]
        # angle_list = ['_'.join(a_list) for a_list in angle_list]
        # datafile_list = [
        #     os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study-3_src-withsir-rirsmall/{}-{}/{}/metrics-makemix_same_00.pkl".format(x[0], x[1], sep_method))
        #     for x in product(t60_list, angle_list)
        # ]
        # concate_ret_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study-3_src-withsir-rirsmall/sep_ret--{}.pkl".format(sep_method))
        # concate_sep_ret(datafile_list, concate_ret_path)

    elif state == 3:
        """ 统计识别正确的场景的SIR [按百分比记] """
        sep_method = 'ILRMA'
        suffix = 'confirm'

        # t60 = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
        # angle_interval = [20, 30, 40, 70, 90, 110]
        # datafile_list = [
        #     os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu_remix/t60_angle_interval_study_withsir/{}-{}/{}".format(x[0], x[1], sep_method))
        #     for x in product(t60, angle_interval)
        # ]

        t60 = [0.16, 0.36, 0.61]
        t60 = ["{:.3f}".format(x) for x in t60]
        datafile_list = [
            os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/{}/{}".format(x, sep_method))
            for x in t60
        ]

        dataname_list = ['makemix_same_00']
        rkg_method = [
            # 'ge2e_withoutaug-cossim',
            # 'ge2e_withaug-cossim',
            # 'xvec_withaug-plda_withaug_onlyln',
            'xvec_sepaug-plda_withaug_onlyln',
            'xvec_diraug-plda_withaug_onlyln',
            'xvec_withoutaug-plda_withoutaug_onlyln',
            # 'xvec_withoutaug-plda_withaug_onlyln',
        ]
        # out_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu_remix/t60_angle_interval_study_withsir/rkg_hist_ret--{}--SDR.".format(sep_method) + suffix + ".pkl")
        out_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/rkg_hist_ret--{}--SDR.".format(sep_method) + suffix + ".pkl")
        get_hist_data(datafile_list, rkg_method, dataname_list, out_path, sub_state=sub_state)

        # out_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/extracted_sep_ret--{}--improved.".format(sep_method) + ".pkl")
        # # out_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu_remix/t60_angle_interval_study_withsir/sep_rkgtrue_ret--{}--improved".format(sep_method) + suffix + '.pkl')
        # get_extracted_metric_data(datafile_list, rkg_method, dataname_list, out_path, sub_state=sub_state, suffix=suffix)
    
    elif state == 4:
        "用于统计表四识别准确率"
        """ 统计所有场景下的识别效果 """
        # sep_method = ['ILRMA', 'MVAE_onehot', 'MVAE_onehot_ilrmainit'][2]
        sep_method = "ILRMA"
        dataname_list = ['makemix_same_00']
        suffix = '0'
        
        # t60 = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
        # angle_interval = [20, 30, 40, 70, 90, 110]
        # datafile_list = [
        #     os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu_remix/t60_angle_interval_study_withsir/{}-{}/{}".format(x[0], x[1], sep_method))
        #     for x in product(t60, angle_interval)
        # ]
        # out_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu_remix/t60_angle_interval_study_withsir/rkg_all_ret--{}.txt".format(sep_method))

        t60 = [0.16, 0.36, 0.61]
        t60 = ["{:.3f}".format(x) for x in t60]
        datafile_list = [
            os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/{}/{}".format(x, sep_method))
            for x in t60
        ]
        out_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/rkg_all_ret--{}.".format(sep_method) + suffix + ".txt")

        rkg_method = [
            # 'ge2e_withoutaug-cossim',
            # 'ge2e_withaug-cossim',
            'xvec_sepaug-plda_withaug_onlyln',
            'xvec_diraug-plda_withaug_onlyln',
            'xvec_withoutaug-plda_withoutaug_onlyln',
            # 'xvec_withoutaug-plda_withaug_onlyln',
        ]
        get_all_rkg_data(datafile_list, rkg_method, dataname_list, out_path, suffix=suffix)

    elif state == 5:
        t60 = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
        angle_interval = [20, 30, 40, 70, 90, 110]
        sep_method = 'ILRMA'
        dataname_list = ['makemix_same_00']
        datafile_list = [
            os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study/{}-{}/{}".format(x[0], x[1], sep_method))
            for x in product(t60, angle_interval)
        ]
        out_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study/sdr2sir--{}--all.pkl".format(sep_method))
        sdr2sir(datafile_list, dataname_list, out_path)

    elif state == 6:
        method = [
            'rkg_ret-makemix_same_00-xvec_withaug-plda_withaug_onlyln',
            # 'rkg_ret-makemix_same_00-xvec_withoutaug-plda_withaug_onlyln',
            # 'rkg_ret-makemix_same_00-xvec_withoutaug-plda_withoutaug_onlyln',
            # 'rkg_ret-makemix_same_00-ge2e_withoutaug-cossim',
            # 'rkg_ret-makemix_same_00-ge2e_withaug-cossim',
        ]
        sep_method = 'ILRMA'
        # t60 = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
        # angle_interval = [20, 30, 40, 70, 90, 110]
        # datafile_list = [
        #     os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study_withsir/{}-{}/{}/{}.pkl".format(x[0], x[1], sep_method, x[2]))
        #     for x in product(t60, angle_interval, method)
        # ]
        # concate_ret_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study_withsir/rkg_ret--{}.pkl".format(sep_method))
        # t60 = [0.16, 0.36, 0.61]
        # t60 = ["{:.3f}".format(x) for x in t60]
        # datafile_list = [
        #     os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study/{}/{}/{}.pkl".format(x[0], sep_method, x[1]))
        #     for x in product(t60, method)
        # ]
        # concate_ret_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study/rkg_ret--{}--xvec.pkl".format(sep_method))
        t60_list = [0.16, 0.36, 0.61]
        t60_list = ['{:.3f}'.format(x) for x in t60_list]
        # angle_list = [[30, 75, 120, 150], [30, 90, 120, 165], [15, 45, 75, 150], [30, 45, 105, 150], [30, 45, 105, 120]]
        angle_list = [[30, 105, 150], [75, 120, 165], [15, 45, 75], [30, 45, 120], [60, 90, 105], [30, 45, 60]]
        angle_list = [[str(x) for x in a_list] for a_list in angle_list]
        angle_list = ['_'.join(a_list) for a_list in angle_list]
        datafile_list = [
            os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study-3_src-withsir-rirsmall/{}-{}/{}/{}.pkl".format(x[0], x[1], sep_method, x[2]))
            for x in product(t60_list, angle_list, method)
        ]
        concate_ret_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study-3_src-withsir-rirsmall/rkg2sir_ret--{}--xvec.txt".format(sep_method))
        rkg2sdr(datafile_list, concate_ret_path)

    elif state == 7:
        t60 = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
        angle_interval = [20, 30, 40, 70, 90, 110]
        sep_method = ['ILRMA', 'MVAE_onehot', 'MVAE_onehot_ilrmainit']
        dataname_list = ['makemix_same_00']
        datafile_list = [
            os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study_withsir/{}-{}/{}".format(x[0], x[1], x[2]))
            for x in product(t60, angle_interval, sep_method)
        ]
        rkg_method = [
            # 'ge2e_withoutaug-cossim',
            # 'ge2e_withaug-cossim',
            'xvec_withaug-plda_withaug_onlyln',
            # 'xvec_withoutaug-plda_withoutaug_onlyln',
            # 'xvec_withoutaug-plda_withaug_onlyln',
        ]
        out_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study_withsir/")
        get_sep_basedon_rkg(datafile_list, dataname_list, rkg_method, out_path)

    elif state == 10:
        datafile_list = [
            os.path.join(script_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study_withsir/0.610/ILRMA/metrics-makemix_same_00.pkl'),
            os.path.join(script_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study_withsir/0.610/MVAE_onehot_ilrmainit/metrics-makemix_same_00.pkl'),
        ]
        forgithubdemo(datafile_list)

    elif state == 11:

        t60 = [0.15, 0.25, 0.35]  #, 0.45, 0.55, 0.65
        # t60 = [0.45, 0.55, 0.65]  #, 
        angle_interval = [20, 30, 40, 70, 90, 110]
        datafile_list = [
            os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu_remix/t60_angle_interval_study_withsir/{}-{}/makemix_same_00.pkl".format(x[0], x[1]))
            for x in product(t60, angle_interval)
        ]
        print("\ndon't forget to adjust makemix！！！！！！！！！\n")

        # t60 = [0.16, 0.36, 0.61]
        # t60 = ["{:.3f}".format(x) for x in t60]
        # datafile_list = [
        #     os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/{}/makemix_same_00.pkl".format(x))
        #     for x in t60
        # ]
        # print("\ndon't forget to adjust makemix！！！！！！！！！\n")

        sep_method = 'ILRMA'
        suffix = '0.'
        rkg_method_list = [
            # 'xvec_sepaug-plda_withaug_onlyln',
            # 'xvec_diraug-plda_withaug_onlyln',
            'xvec_withoutaug-plda_withoutaug_onlyln',
        ]
        # sep_suffix_list = [10, 15, 20, 25]
        sep_suffix_list = ['']
        for sep_suffix in sep_suffix_list:
            get_extracted_metrics(datafile_list, sep_method, suffix=suffix, rkg_method_list=rkg_method_list, fs=16000, sub_state=sub_state, sep_suffix=sep_suffix)

    elif state == 12:
        """ 统计识别正确的场景的SIR [按百分比记] """
        sep_method = 'ILRMA'
        # t60 = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
        # angle_interval = [20, 30, 40, 70, 90, 110]
        # datafile_list = [
        #     os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu_remix/t60_angle_interval_study_withsir/{}-{}/{}".format(x[0], x[1], sep_method))
        #     for x in product(t60, angle_interval)
        # ]

        t60 = [0.16, 0.36, 0.61]
        t60 = ["{:.3f}".format(x) for x in t60]
        datafile_list = [
            os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/{}/{}".format(x, sep_method))
            for x in t60
        ]

        dataname_list = ['makemix_same_00']
        rkg_method = [
            # 'ge2e_withoutaug-cossim',
            # 'ge2e_withaug-cossim',
            # 'xvec_withaug-plda_withaug_onlyln',
            'xvec_sepaug-plda_withaug_onlyln',
            'xvec_diraug-plda_withaug_onlyln',
            'xvec_withoutaug-plda_withoutaug_onlyln',
            # 'xvec_withoutaug-plda_withaug_onlyln',
        ]
        # out_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu_remix/t60_angle_interval_study_withsir/extracted_hist_ret--{}--SDR.0.pkl".format(sep_method))
        out_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/extracted_hist_ret--{}--SDR.confirm.pkl".format(sep_method))
        get_extracted_hist_data(datafile_list, rkg_method, dataname_list, out_path, sub_state=sub_state)


    elif state == 13:
        suffix = '0'
        sep_method = 'ILRMA'
        sep_suffix_list = ['10s.', '15s.', '20s.', '25s.', '']
        # sep_suffix_list = ['']
        for sep_suffix in sep_suffix_list:
            t60 = [0.16]
            t60 = ["{:.3f}".format(x) for x in t60]
            datafile_path = [
                os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/{}/{}".format(x, sep_method))
                for x in t60
            ]
        rkg_method = [
            'xvec_sepaug-plda_withaug_onlyln',
            'xvec_diraug-plda_withaug_onlyln',
        ]
        dataname = 'makemix_same_00'
        out_path = os.path.join(script_prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/sep_ret_len_study--{}.".format(sep_method) + suffix + ".pkl")
        get_different_len_data(datafile_path, rkg_method, dataname, out_path, sep_suffix_list, sub_state=0)
        

if __name__ == "__main__":
    main(state=12, sub_state=0, script_prefix=hp.path.script_prefix)
     


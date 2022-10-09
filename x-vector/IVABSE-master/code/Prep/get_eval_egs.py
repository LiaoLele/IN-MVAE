import librosa
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.realpath(os.path.dirname(__file__)))))
import torch
import datetime
import pickle
from itertools import combinations, product
from collections import defaultdict, Counter
import random
import numpy as np
import librosa
import scipy.io as sio
import soundfile as sf
from scipy.signal import convolve
import mir_eval
from Network.SepNet import vae
from Network.SpkrNet.ge2e import speaker_encoder_ge2e
from SepAlgo.ilrma import myilrma, ilrma
from SepAlgo.gmm import avgmm
from SepAlgo.Aux_chainlike import chainlike, chainlike_prob
from SepAlgo.mvae import mvae_onehot, mvae_ge2e, mvae_onehot_official
from utils import mysort, zero_pad, assign_rir, assign_sir
# from pyroomacoustics.bss.ilrma import ilrma


def make_src_signal(data_path, out_path, enroll_len=60, utt_len=5, fs=16000):
    """ Split concatenated signal from data_path to utterances as source signal for creating mixtures in evaluation stage """
    """ 
    Args:
        `data_path`: [str] path where test concatenated signals are saved
        `out_path`: [str] path where test_signal_info.txt and test_signal_spk2utt.pkl will be saved
        `enroll_len`: [int][in second] enrollment signal length, always saved [0: enroll_len * fs] for enrollment
        `utt_len`: [int][in second] mixture length
        `fs`: [int] sampling rate
    Out:
        `test_signal_info.txt`: txt file that saves enroll_len and utt_len for review
        `spk2utt`: [dict][Not returned but saved] dict that saves all the utterances
                    {'speakerxxxx': [(absolute-path-to-wav, offset, dur), (), ...], '': [], ...}
        `test_signal_spk2utt.pkl`: pickle file that saves spk2utt
    """
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(data_path, 'info.pkl'), 'rb') as f:
        info = pickle.load(f)
    f_txt = open(os.path.join(out_path, 'test_signal_info.txt'), 'wt')
    print("Enrollment length preserved is {} seconds-long".format(enroll_len), file=f_txt)
    print("Each source signal is {} second in length".format(utt_len), file=f_txt)
    print("\n", file=f_txt)

    spk2utt = {}    
    enroll_len = int(enroll_len * fs)
    utt_len = int(utt_len * fs)
    for src_path, dur, _ in info:
        spkr_name = os.path.basename(src_path).split('.')[0].rsplit('-', 1)[1]
        if spkr_name not in spk2utt:
            spk2utt[spkr_name] = []
        num_utt = (dur - enroll_len) // utt_len
        for i in range(num_utt):
            spk2utt[spkr_name].append((i, src_path, enroll_len + i * utt_len, utt_len))
    for spkr in spk2utt.keys():
        print("{}: {} utterances".format(spkr, len(spk2utt[spkr])), file=f_txt)
    
    with open(os.path.join(out_path, 'test_signal_spk2utt.pkl'), 'wb') as f:
        pickle.dump(spk2utt, f)
    f_txt.close()


# def MakePair(srcinfo_path, out_dir, src_num=2, mode='print_avaliable_pair_number', 
#              n_pair=None, n_egs_per_pair=None,
#              use_all_speakers=True, subset_func=None,
#              fs=16000, cols=None,
#              seed=None, suffix=''):
#     """ Make mixpairs """
#     """ WITH RANDOM
#     Args:
#         srcinfo_path: [str] path where test_signal_spk2utt.pkl is saved
#         out_path: [str] path where mix2pair will be saved
#         src_num: [int] number of speakers in each mixture
#         print_total_pair_num: [bool] if True, the method outputs total number of speaker pairs and return
#         n_pair, n_egs_per_pair: [int] if add_mode is True, means total value including those from orifile_path
#         use_all_speakers: [bool] If True, all speakers in srcinfo_path must appear
    
#     Out:
#         makemix: [dict]
#                  [[(absolute-path-to-wav, offset, dur), (absolute-path-to-wav, offset, dur)], [(), ()], ...]
#     """

#     with open(srcinfo_path, 'rb') as f:
#         spk2utt = pickle.load(f)
#     spkr_list = list(spk2utt.keys())
#     spkr_pair = list(combinations(spkr_list, src_num))
#     if mode == 'print_avaliable_pair_number':
#         print("Total number of available pairs is {}".format(len(spkr_pair)))
#         return

#     if seed is None:
#         seed = datetime.datetime.now()
#     random.seed(seed)
#     os.makedirs(out_dir, exist_ok=True)
#     out_basename = os.path.basename(srcinfo_path).split('.')[1: -1]
#     f_readme = open(os.path.join(out_dir, '.'.join(['Readme-MakePair'] + out_basename + [suffix, 'txt'])), 'wt')
#     print("The sources are from {}".format(srcinfo_path), file=f_readme)
#     print("Number of sources in each pair: {}".format(src_num), file=f_readme)
#     print("Number of speaker pairs: {}".format(n_pair), file=f_readme)
#     print("Number of utterances per speaker pair: {}".format(n_egs_per_pair), file=f_readme)
#     print("Whether to use all speakers: {}".format(use_all_speakers), file=f_readme)

#     mix2pair = [[] for _ in range(src_num)]
#     if len(spkr_pair) < n_pair:
#         raise ValueError("n_pair exceeds the total number of available pairs--{}, you may need to reconsider n_pair and n_egs_per_pair".format(len(spkr_pair)))
#     random.shuffle(spkr_pair)
#     if use_all_speakers:
#         if n_pair * src_num < len(spkr_list):
#             real_used_spkr_num = int(n_pair * src_num)
#             print(f"n_pair is too small to use all speakers, therefore use {real_used_spkr_num} speakers instead!", file=f_readme)
#             print(f"n_pair is too small to use all speakers, therefore use {real_used_spkr_num} speakers instead!")
#         else:
#             real_used_spkr_num = len(spkr_list)
#         vis_spkr_dict = defaultdict(list)
#         vis_spkr_list = []
#         num_generated_pair = 0
#         spkr_pair_idx = 0
#         while len(set(vis_spkr_list)) < real_used_spkr_num or num_generated_pair < n_pair:
#             if num_generated_pair < n_pair:
#                 """ generate anyway if num_generated_pair is less than n_pair """
#                 num_generated_pair += 1
#                 for spkr in spkr_pair[spkr_pair_idx]:
#                     vis_spkr_dict[spkr].append(spkr_pair[spkr_pair_idx])
#                     vis_spkr_list.append(spkr)
#                 spkr_pair_idx += 1
#             else:
#                 """ if num_generated_pair == n_pair but some speakers are not seen """
#                 # find the most frequently used speaker
#                 temp = Counter(vis_spkr_list)
#                 spkr_most_common = temp.most_common(1)[0][0]
#                 spkrs_del = vis_spkr_dict[spkr_most_common][-1]
#                 for spkr_del in spkrs_del:
#                     vis_spkr_dict[spkr_del].remove(spkrs_del)
#                     vis_spkr_list.remove(spkr_del)
#                 num_generated_pair -= 1
#         fin_pair = set()
#         for spkr in vis_spkr_dict.keys():
#             for val in vis_spkr_dict[spkr]:
#                 fin_pair.add(val)
#         assert len(fin_pair) == n_pair
#         assert len(set(vis_spkr_list)) == real_used_spkr_num
#     else:
#         fin_pair = spkr_pair[0: n_pair]

#     pair2utts, pair2totaluttnum = get_pair2utts(fin_pair, spk2utt)
#     required_uttnum = int(n_pair * n_egs_per_pair)
#     if sum(list(pair2totaluttnum.values())) < required_uttnum:
#         raise ValueError(f"Required number of utterance {required_uttnum} exceeds total number of utterance available {sum(pair2totaluttnum)}! Please reconsider!")
#     pair2uttnum = defaultdict(int)
#     num_generated_utts = 0
#     while num_generated_utts < required_uttnum:
#         for spkrs in pair2totaluttnum.keys():
#             if pair2uttnum[spkrs] < pair2totaluttnum[spkrs]:
#                 pair2uttnum[spkrs] += 1
#                 num_generated_utts += 1
#                 if num_generated_utts >= required_uttnum:
#                     break
#     for spkrs, uttids in pair2utts.items():
#         uttnum = pair2uttnum[spkrs]
#         chosen_uttids = random.sample(uttids, uttnum)
#         for uttid in chosen_uttids:
#             for i in range(src_num):
#                 mix2pair[i].append(list(spk2utt[spkrs[i]][uttid[i]]) + [fs])
#     out_name = os.path.join(out_dir, '.'.join(['Pair'] + out_basename + [suffix, 'xls']))
#     print(f"Pair information saved in {out_name}", file=f_readme)
#     writer = pd.ExcelWriter(out_name)
#     for i in range(src_num):
#         df = pd.DataFrame(mix2pair[i], columns=cols)
#         df.to_excel(writer, sheet_name=str(i), index_label='Index')
#     writer.save()
#     writer.close()


def make_mix2pair(srcinfo_path, out_path, src_num=2, print_total_pair_num=False, n_pair=None, n_egs_per_pair=None, use_all_speakers=True):
    """ Make mixpairs """
    """ WITH RANDOM
    Args:
        srcinfo_path: [str] path where test_signal_spk2utt.pkl is saved
        out_path: [str] path where mix2pair will be saved
        src_num: [int] number of speakers in each mixture
        print_total_pair_num: [bool] if True, the method outputs total number of speaker pairs and return
        n_pair, n_egs_per_pair: [int] if add_mode is True, means total value including those from orifile_path
        use_all_speakers: [bool] If True, all speakers in srcinfo_path must appear
    
    Out:
        makemix: [dict]
                 [[(absolute-path-to-wav, offset, dur), (absolute-path-to-wav, offset, dur)], [(), ()], ...]
    """
    random.seed(datetime.datetime.now())
    os.makedirs(out_path, exist_ok=True)
    f_txt = open(os.path.join(out_path, 'pair_info_read.txt'), 'wt')
    print("SRC_PATH: {}".format(srcinfo_path), file=f_txt)
    print("SRC_NUM_PER_MIXTURE: {}".format(src_num), file=f_txt)
    print("N_PAIR: {}".format(n_pair), file=f_txt)
    print("N_EGS_PER_PAIR: {}".format(n_egs_per_pair), file=f_txt)
    print("USE ALL SPEAKERS: {}".format(use_all_speakers), file=f_txt)
    print("\n", end='', file=f_txt)

    with open(srcinfo_path, 'rb') as f:
        spk2utt = pickle.load(f)
    spkr_list = list(spk2utt.keys())
    spkr_pair = list(combinations(spkr_list, src_num))
    if print_total_pair_num:
        print("Total number of available pairs is {}".format(len(spkr_pair)))
        return
    mix2pair = []
    idx = 1

    # i.e. Create pairs using mode "n_egs_per_pair egs from n_pair pairs"
    if len(spkr_pair) < n_pair:
        raise ValueError("n_pair exceeds the total number of available pairs--{}, you may need to reconsider n_pair and n_egs_per_pair".format(len(spkr_pair)))
    random.shuffle(spkr_pair)

    if use_all_speakers:
        assert n_pair * 2 >= len(spkr_list), 'n_pair is too small to use all the speakers'
        vis_spkr_dict = defaultdict(list)
        vis_spkr_list = []
        num_generated_pair = 0
        spkr_pair_idx = 0
        while len(vis_spkr_dict.keys()) < len(spkr_list) or num_generated_pair < n_pair:
            if num_generated_pair < n_pair:
                num_generated_pair += 1
                for spkr in spkr_pair[spkr_pair_idx]:
                    vis_spkr_dict[spkr].append(spkr_pair[spkr_pair_idx])
                    vis_spkr_list.append(spkr)
                spkr_pair_idx += 1
            else:
                temp = Counter(vis_spkr_list)
                # spkr_most_common = temp.most_common(1)
                # for spkr_del in vis_spkr_dict[spkr_most_common]:
                spkr_most_common = temp.most_common(1)[0][0]
                spkrs_del = vis_spkr_dict[spkr_most_common][-1]
                for spkr_del in spkrs_del:
                    vis_spkr_dict[spkr_del].remove(vis_spkr_dict[spkr_most_common])
                    vis_spkr_list.remove[spkr_del]
                num_generated_pair -= 1
        fin_pair = set()
        for spkr in vis_spkr_dict:
            for val in vis_spkr_dict[spkr]:
                fin_pair.add(val)
        assert len(fin_pair) == n_pair
    else:
        fin_pair = spkr_pair[0: n_pair]

    for cnt, spkr_name in enumerate(fin_pair):
        print(cnt)
        _, spkr_name = mysort(spkr_name)
        utt_pair = list(product(*[range(len(spk2utt[spkr_name[i]])) for i in range(len(spkr_name))]))
        utt_pair_num = len(utt_pair)
        if utt_pair_num < n_egs_per_pair:
            print("Total utterance combination for speakers: {} is less than n_egs_per_pair".format(spkr_name))
        utt_select = random.sample(utt_pair, n_egs_per_pair)
        for utt_id_all in utt_select:
            mix = [tuple([spk2utt[spkr_name[i]][utt_id_all[i]] for i in range(len(spkr_name))])]
            mix2pair.append(mix)
            print("{:>04d}: {}".format(idx, mix), file=f_txt)
            idx += 1
    with open(os.path.join(out_path, 'pair.pkl'), 'wb') as f:
        pickle.dump(mix2pair, f)
    f_txt.close()


def get_seped_utter(rirdata_path, pairinfo_path, out_path, suffix_list=None, sub_state=0, method='ILRMA', prefix=None, rep_metrics_path=None, **kwargs):
    """ separate mixtures in makemix.n.pkl and save separated utterances to out_path """
    """
    Args:
        `rirdata_path`: [str] path where idx2rir.pkl is saved
        `pairinfo_path`: [list] path where makemix.pkl is saved
        `out_path`: [list] path where .wav file will be saved
        `sub_state`: [int] 0 is execution state; 1 is debugging state
        `method`: [str] separation method "ILRMA" and "MVAE"
        `kwargs`: [dict] params for STFT and method
    Out:
        `metric_dict`: [Not returned but saved][dict]
                    dict object that saves SDR, SIR, SAR information of separated data in makemix.n.pkl order
                    {'mix': [makemix[0], makemix[1], ...], 'SDR': {'ori': [], 'sep': []}, 'SIR': {'ori': [], 'sep': []}, 'SAR': {'ori': [], 'sep': []}}
        `metrics.n.pkl`: pickle file that saves metric_dict for job n
        `log.n.txt`: txt file that saves log for job n
                     each line of log is 'Finish processing index m for job n' 
        `sep`: saved in wav file
    """
    print(sub_state)
    f_idx2rir = open(os.path.join(rirdata_path, 'idx2rir.pkl'), 'rb')
    idx2rir = pickle.load(f_idx2rir)
    f_idx2rir.close()
    nfft = int(kwargs['fs'] * kwargs['stft_len'])
    hop = int(kwargs['fs'] * kwargs['stft_hop'])
    fs = kwargs['fs']

    # if method.startswith('MVAE_onehot'):
    #     modelfile_path = kwargs['vae_model_path']
    #     n_embedding = kwargs['embedding_dim']
    #     device = kwargs['device']
    #     model = vae.net(n_embedding)
    #     model.load_state_dict(torch.load(modelfile_path, map_location=device))
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     model.to(device)
    #     model.eval()
    # elif method == 'MVAE_ge2e':
    #     fb_mat = torch.from_numpy(librosa.filters.mel(16000, nfft, n_mels=40)).unsqueeze(0)
    #     vaemodel_path = kwargs['vae_model_path']
    #     spkrmodel_path = kwargs['spkr_model_path']
    #     n_embedding = kwargs['embedding_dim']
    #     device = kwargs['device']
    #     vae_model = vae.net(n_embedding)
    #     vae_model.load_state_dict(torch.load(vaemodel_path, map_location=device))
    #     vae_model.to(device)
    #     vae_model.eval()
    #     spkr_model = speaker_encoder_ge2e()
    #     spkr_model_dict = spkr_model.state_dict()
    #     pretrained_dict = torch.load(spkrmodel_path, map_location=torch.device('cpu'))
    #     pretrained_dict_rename = {}
    #     for k, v in pretrained_dict.items():
    #         try:
    #             param_name = k.split('.', 1)[1]
    #             pretrained_dict_rename[param_name] = v
    #         except IndexError:
    #             pass
    #     pretrained_dict_rename = {k: v for k, v in pretrained_dict_rename.items() if k in spkr_model_dict}
    #     spkr_model_dict.update(pretrained_dict_rename)
    #     spkr_model.load_state_dict(spkr_model_dict)
    #     spkr_model.cuda(device)
    #     spkr_model.eval()
    
    for path_idx, pairinfo in enumerate(pairinfo_path):
        print('Current makemix is {}'.format(pairinfo))
        os.makedirs(out_path[path_idx], exist_ok=True)
        # src_len = int(suffix_list[path_idx] * fs)
        # suffix = str(suffix_list[path_idx]) + 's'
        # print(suffix)
        suffix = ''
        # if sub_state == 0:
        #     if os.path.exists(os.path.join(out_path[path_idx], 'log-{}.txt'.format(os.path.basename(pairinfo[0: -4])))):
        #         os.system("mv {} {}".format(os.path.join(out_path[path_idx], 'log-{}.txt'.format(os.path.basename(pairinfo[0: -4]))), os.path.join(out_path[path_idx], 'log-{}.bak.txt'.format(os.path.basename(pairinfo[0: -4])))))
        #     if os.path.exists(os.path.join(out_path[path_idx], 'metrics-{}.pkl'.format(os.path.basename(pairinfo[0: -4])))):
        #         os.system("mv {} {}".format(os.path.join(out_path[path_idx], 'metrics-{}.pkl'.format(os.path.basename(pairinfo[0: -4]))), os.path.join(out_path[path_idx], 'metrics-{}.bak.pkl'.format(os.path.basename(pairinfo[0: -4])))))
        
        if rep_metrics_path is not None:
            with open(rep_metrics_path[path_idx], 'rb') as f:
                rep_metrics = pickle.load(f)

        f_makemix = open(pairinfo, 'rb')
        makemix = pickle.load(f_makemix)
        f_makemix.close()
        metric_dict = {'mix': [], 'SDR': {'ori': [], 'sep': []}, 'SIR': {'ori': [], 'sep': []}, 'SAR': {'ori': [], 'sep': []}, 'PERM': {'sep': []}}

        np.random.seed(0)
        torch.manual_seed(0)
        for idx, (((spkr_id_1, srcdata_path_1, offset_1, duration_1), (spkr_id_2, srcdata_path_2, offset_2, duration_2)), gidx_rir, mix_sir) in enumerate(makemix):
            # print("Processing {}/{} mixture.".format(idx + 1, len(makemix)))
            if mix_sir == 0 and rep_metrics_path is not None:
                assert len(metric_dict['mix']) == idx
                assert makemix[idx][0: -1] == rep_metrics['mix'][idx]
                if sub_state == 0:
                    metric_dict['mix'].append(makemix[idx])
                    for i, m in enumerate(['SDR', 'SIR', 'SAR']):
                        metric_dict[m]['sep'].extend(rep_metrics[m]['sep'][idx * 2: idx * 2 + 2])
                        metric_dict[m]['ori'].extend(rep_metrics[m]['ori'][idx * 2: idx * 2 + 2])
                    metric_dict['PERM']['sep'].append(rep_metrics["PERM"]['sep'][idx])

                    src_name_1 = os.path.basename(srcdata_path_1)[0: -4]
                    src_name_2 = os.path.basename(srcdata_path_2)[0: -4]
                    out_path_ori = os.path.join(os.path.dirname(rep_metrics_path[path_idx]), "{}-{}_{}-{}_{}".format(src_name_1, spkr_id_1, src_name_2, spkr_id_2, method))
                    out_path_necessary = os.path.join(out_path[path_idx], "{}-{}_{}-{}".format(src_name_1, spkr_id_1, src_name_2, spkr_id_2))
                    os.system("cp {} {}".format(out_path_ori + '_sep.wav', out_path_necessary + '_sep.wav'))
                    with open(os.path.join(out_path[path_idx], 'log-{}.txt'.format(os.path.basename(pairinfo[0: -4]))), 'at') as f_log:
                        print('Finish processing index {} for {}'.format(idx, os.path.basename(pairinfo[0: -4])), file=f_log)
            else:
                # Generate mixture and source signals
                src_name_1 = os.path.basename(srcdata_path_1).rsplit('.', 1)[0].rsplit('-', 1)[1]
                src_name_2 = os.path.basename(srcdata_path_2).rsplit('.', 1)[0].rsplit('-', 1)[1]
                # src_name_1 = os.path.basename(srcdata_path_1)[0: -4]
                # src_name_2 = os.path.basename(srcdata_path_2)[0: -4]
                out_path_necessary = os.path.join(out_path[path_idx], "{}-{}_{}-{}".format(src_name_1, spkr_id_1, src_name_2, spkr_id_2))
                if prefix is not None:
                    srcdata_path_1 = os.path.join(prefix, srcdata_path_1)
                    srcdata_path_2 = os.path.join(prefix, srcdata_path_2)
                    # srcdata_path_1 = os.path.join(prefix, 'DATASET/Librispeech/concatenate/test_clean/' + src_name_1 + '.wav')
                    # srcdata_path_2 = os.path.join(prefix, 'DATASET/Librispeech/concatenate/test_clean/' + src_name_2 + '.wav')
                src_1, _ = sf.read(srcdata_path_1, start=offset_1, stop=offset_1 + duration_1)
                src_2, _ = sf.read(srcdata_path_2, start=offset_2, stop=offset_2 + duration_2)
                src_1 = src_1 / np.max(np.abs(src_1))
                src_2 = src_2 * np.std(src_1) / np.std(src_2)
                assert src_1.shape[0] == src_2.shape[0]
                src = np.stack((src_1, src_2), axis=1)
                sf.write(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(out_path_necessary))), os.path.basename(out_path_necessary) + '_src.' + suffix + 'wav'), src, fs)
                # if src_len < src.shape[0]:
                #     src = src[0: src_len, :]

                """ if the rir is simulated """
                # rir = idx2rir[gidx_rir][1]  
                # # mic-1
                # mix_11 = convolve(src_1, rir[0][0])
                # mix_21 = convolve(src_2, rir[0][1])
                # # compute sir factor
                # scale_fac = np.std(mix_11) * 10**(-mix_sir / 20) / np.std(mix_21)
                # mix_21 = convolve(src_2 * scale_fac, rir[0][1])
                # # mic-2
                # mix_12 = convolve(src_1, rir[1][0])
                # mix_22 = convolve(src_2 * scale_fac, rir[1][1])
                # # sum
                # mix_1 = mix_11 + mix_21
                # mix_2 = mix_12 + mix_22

                """ if the rir is real """
                # rir_path_1, rir_path_2 = idx2rir[gidx_rir]['path']
                # channel = idx2rir[gidx_rir]['channel']
                # rir_path_1 = os.path.join(prefix, rir_path_1)
                # rir_path_2 = os.path.join(prefix, rir_path_2)
                # rir_1 = sio.loadmat(rir_path_1)
                # rir_2 = sio.loadmat(rir_path_2)
                # rir_1 = rir_1['impulse_response'][:, channel].T
                # rir_2 = rir_2['impulse_response'][:, channel].T
                # rir_1 = librosa.core.resample(np.asfortranarray(rir_1), 48000, 16000)
                # rir_2 = librosa.core.resample(np.asfortranarray(rir_2), 48000, 16000)
                # rir_1 = rir_1[:, 0: int(0.8 * 16000)]
                # rir_2 = rir_2[:, 0: int(0.8 * 16000)]
                # # max_l = max(rir_1.shape[-1], rir_2.shape[-1])
                # # rir_1 = rir_1[:, 0: max_l]
                # # rir_2 = rir_2[:, 0: max_l]

                # mix_11 = convolve(src_1, rir_1[0, :])
                # mix_21 = convolve(src_2, rir_2[0, :])
                # scale_fac = np.std(mix_11) * 10**(-mix_sir / 20) / np.std(mix_21)
                # mix_1 = convolve(src_1, rir_1[0, :]) + convolve(src_2 * scale_fac, rir_2[0, :])
                # mix_2 = convolve(src_1, rir_1[1, :]) + convolve(src_2 * scale_fac, rir_2[1, :])

                # mix = np.stack((mix_1, mix_2), axis=1)
                # mix = mix[0: src.shape[0], :]
                # mix = mix / np.max(np.abs(mix))
                # # if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(out_path_necessary)), os.path.basename(out_path_necessary) + '_mix.' + suffix + '.wav')):
                # sf.write(os.path.join(os.path.dirname(os.path.dirname(out_path_necessary)), os.path.basename(out_path_necessary) + '_mix.' + suffix + 'wav'), mix, fs)
                # mix, _ = sf.read(os.path.join(os.path.dirname(os.path.dirname(out_path_necessary)), os.path.basename(out_path_necessary) + '_mix.' + suffix + '.wav'))
                # sf.write(out_path_necessary + '_srctest.wav', src, fs)
                # if method.startswith("MVAE"):
                #     src = zero_pad(src.T, 4, hop_length=hop)
                #     mix = zero_pad(mix.T, 4, hop_length=hop)
                #     src, mix = src.T, mix.T

                # # Separate mixture using method
                # mix_spec = [librosa.core.stft(np.asfortranarray(mix[:, ch]), n_fft=nfft, hop_length=hop, win_length=nfft) for ch in range(mix.shape[1])]
                # mix_spec = np.stack(mix_spec, axis=1)
                # if method == 'ILRMA':
                #     sep_spec, flag = myilrma(mix_spec, 1000, n_basis=2)
                # elif method == 'GMM':
                #     sep_spec, flag = avgmm(mix_spec, 1000, state_num=2)
                # elif method == "MVAE_onehot":
                #     sep_spec, flag = mvae_onehot(mix_spec.swapaxes(1, 2), model, n_iter=1000, device=device, ilrma_init=False)
                # elif method == 'MVAE_onehot_ilrmainit':
                #     sep_spec, flag = mvae_onehot(mix_spec.swapaxes(1, 2), model, n_iter=1000, device=device, ilrma_init=True)
                # elif method == "MVAE_ge2e":
                #     sep_spec, flag = mvae_ge2e(mix_spec.swapaxes(1, 2), vae_model, spkr_model, fb_mat=fb_mat, n_iter=1000, device=device)
                # # elif method == 'MVAE_onehot_official_ilrmainit':
                # #     sep_spec, flag = mvae_onehot_official(mix_spec.swapaxes(1, 2), model, n_iter=1000, device=device, ilrma_init=True)
                # # elif method == 'MVAE_onehot_official':
                # #     sep_spec, flag = mvae_onehot_official(mix_spec.swapaxes(1, 2), model, n_iter=1000, device=device, ilrma_init=False)

                # sep = [librosa.core.istft(sep_spec[:, ch, :], hop_length=hop, length=src.shape[0]) for ch in range(sep_spec.shape[1])]
                # sep = np.stack(sep, axis=1)
                # metrics = mir_eval.separation.bss_eval_sources(src.T, sep.T)
                # metrics_ori = mir_eval.separation.bss_eval_sources(src.T, mix.T)

                # metric_dict['mix'].append(makemix[idx])
                # for i, m in enumerate(['SDR', 'SIR', 'SAR']):
                #     metric_dict[m]['sep'].extend(metrics[i].tolist())
                #     metric_dict[m]['ori'].extend(metrics_ori[i].tolist())
                # metric_dict['PERM']['sep'].append(metrics[-1])
                
        #         if sub_state == 0:
        #             sf.write(out_path_necessary + '_sep.' + suffix + '.wav', sep, fs)
        #             with open(os.path.join(out_path[path_idx], 'log-{}.'.format(os.path.basename(pairinfo[0: -4])) + suffix + '.txt'), 'at') as f_log:
        #                 print('Finish processing index {} for {}'.format(idx, os.path.basename(pairinfo[0: -4])), file=f_log)
        #             if flag:
        #                 import ipdb; ipdb.set_trace()
        #                 with open(os.path.join(out_path[path_idx], 'anomaly-log-{}.'.format(os.path.basename(pairinfo[0: -4])) + suffix + '.txt'), 'at') as f:
        #                     print(os.path.basename(out_path_necessary), file=f)
        #         # if sub_state == 2:
        #         #     sf.write(out_path_necessary + '_mixtest.wav', mix, fs)
        #         #     sf.write(out_path_necessary + '_srctest.wav', src, fs)
        #         #     sf.write(out_path_necessary + '_septest.wav', sep, fs)
        # if sub_state == 0:
        #     f_metrics = open(os.path.join(out_path[path_idx], 'metrics-{}.'.format(os.path.basename(pairinfo[0: -4])) + suffix + '.pkl'), 'wb')
        #     pickle.dump(metric_dict, f_metrics)
        #     f_metrics.close()
        

def main(state=0, sub_state=0, prefix=None):
    if state == 0:
        """ split concatenated test signal into small segments and obtain test_signal_spk2utt.pkl """
        data_path = prefix + '/DATASET/Librispeech/concatenate_MN/test_dev_clean'
        out_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/SOURCE/length30s'
        enroll_len = 60
        utt_len = 30
        if sub_state == 0:
            make_src_signal(data_path, out_path, enroll_len=enroll_len, utt_len=utt_len, fs=16000)

    elif state == 1:
        """ Create mixture pairs for testing, i.e. mix2pair.pkl """
        srcinfo_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/SOURCE/length30s/test_signal_spk2utt.pkl'
        outfile_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu_remix/t60_angle_interval_study_withsir'
        n_pair = 360
        n_egs_per_pair = 1
        if sub_state == 0:
            make_mix2pair(srcinfo_path, outfile_path, n_pair=n_pair, n_egs_per_pair=n_egs_per_pair, src_num=2, use_all_speakers=True)
        elif sub_state == 1:
            """ Find total number of speaker pairs """
            make_mix2pair(srcinfo_path, outfile_path, print_total_pair_num=True)
        elif sub_state == 2:
            with open(outfile_path, 'rb') as f:
                mix2pair = pickle.load(f)
                spkr_set = set()
                for spkr1, spkr2 in mix2pair:
                    spkr_name1 = os.path.basename(spkr1[1])
                    spkr_name2 = os.path.basename(spkr2[1])
                    spkr_set.add(spkr_name1)
                    spkr_set.add(spkr_name2)
                dummy = 1
            
    elif state == 2:
        """ Assign rir to mix2pair """
        rirfile_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/RIR/test_clean/t60_angle_interval_study/key2gidx.pkl'
        targetfile_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu_remix/t60_angle_interval_study_withsir/pair.pkl'
        t60_list = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
        angle_list = [20, 30, 40, 70, 90, 110]
        # t60_list = [0.16, 0.36, 0.61]
        # t60_list = ['{:.3f}'.format(x) for x in t60_list]
        # assign_key = t60_list.copy()
        # # angle_list = [[30, 75, 120, 150], [30, 90, 120, 165], [15, 45, 75, 150], [30, 45, 105, 150], [30, 45, 105, 120]]
        # # angle_list = [[30, 105, 150], [75, 120, 165], [45, 90, 135], [30, 45, 150]]
        # angle_list = [[90, 30, 150]]  #, [75, 120, 165], [15, 45, 75], [30, 45, 120], [60, 90, 105], [30, 90, 150], [30, 45, 60]]
        # angle_list = [[str(x) for x in a_list] for a_list in angle_list]
        # angle_list = ['_'.join(a_list) for a_list in angle_list]
        assign_key = ["{}-{}".format(t60, angle) for t60, angle in product(t60_list, angle_list)]
        for key in assign_key:
            outfile_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu_remix/t60_angle_interval_study_withsir/' + key + '/makemix_same_00.pkl'
            if sub_state == 0:
                assign_rir(rirfile_path, targetfile_path, outfile_path, assign_key=key, mode=0)
            if sub_state == 1:
                """ Check keys """
                with open(rirfile_path, 'rb') as f:
                    key2gidx = pickle.load(f)
                print(key2gidx.keys())
    
    elif state == 3:
        """ Assign sir to mix2pair """
        t60 = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
        angle_interval = [20, 30, 40, 70, 90, 110]
        targetfile_path = [
            os.path.join(prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu_remix/t60_angle_interval_study_withsir/{}-{}/makemix_same_00.pkl".format(x[0], x[1]))
            for x in product(t60, angle_interval)
        ]
        outfile_path = [
            os.path.join(prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu_remix/t60_angle_interval_study_withsir/{}-{}/makemix_same_00.pkl".format(x[0], x[1]))
            for x in product(t60, angle_interval)
        ]
        # t60 = [0.16, 0.36, 0.61]
        # t60 = ["{:.3f}".format(x) for x in t60]
        # targetfile_path = [
        #     os.path.join(prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/{}/makemix_same_00.pkl".format(x))
        #     for x in t60
        # ]
        # outfile_path = [
        #     os.path.join(prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/{}/makemix_same_00.pkl".format(x))
        #     for x in t60
        # ]
        # t60_list = [0.16, 0.36, 0.61]
        # t60_list = ['{:.3f}'.format(x) for x in t60_list]
        # # angle_list = [[30, 75, 120, 150], [30, 90, 120, 165], [15, 45, 75, 150], [30, 45, 105, 150], [30, 45, 105, 120]]
        # # angle_list = [[30, 105, 150], [75, 120, 165], [45, 90, 135], [30, 45, 150]]
        # angle_list = [[30, 105, 150], [75, 120, 165], [15, 45, 75], [30, 45, 120], [60, 90, 105], [30, 90, 150], [30, 45, 60]]
        # angle_list = [[str(x) for x in a_list] for a_list in angle_list]
        # angle_list = ['_'.join(a_list) for a_list in angle_list]
        # targetfile_path = [
        #     os.path.join(prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study-3_src-withsir-rirsmall-2/{}-{}/makemix_same_00.pkl".format(x[0], x[1]))
        #     for x in product(t60_list, angle_list)
        # ]
        # outfile_path = [
        #     os.path.join(prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study-3_src-withsir-rirsmall-2/{}-{}/makemix_same_00.pkl".format(x[0], x[1]))
        #     for x in product(t60_list, angle_list)
        # ]
        sir_range = [0, -5, 5]
        for i, target_path in enumerate(targetfile_path):
            os.system("cp {} {}".format(target_path, target_path[0: -4] + '.withoutsir.pkl'))
            assign_sir(target_path, outfile_path[i], sir_range, use_random=False, make_order=False)

    elif state == 4:
        """ Using separation algorithms to get separated signals """
        method = "MVAE_onehot"
        # rirdata_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/RIR/test_clean/t60_angle_interval_study/'
        rirdata_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/RIR/test_real_remix/'
        # rirdata_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/RIR/test_real/'
        suffix_list = ['']
        pairinfo_path = [
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study_withsir/0.65-20/makemix_same_00.pkl',
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study_withsir/0.65-30/makemix_same_00.pkl',
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study_withsir/0.65-40/makemix_same_00.pkl',
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study_withsir/0.65-70/makemix_same_00.pkl',
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study_withsir/0.65-90/makemix_same_00.pkl',
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study_withsir/0.65-110/makemix_same_00.pkl',
            prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/0.160/makemix_same_00.pkl',
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/0.160/makemix_same_00.pkl',
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/0.160/makemix_same_00.pkl',
        ]
        out_path = [
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study_withsir/0.65-20/' + method,
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study_withsir/0.65-30/' + method,
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study_withsir/0.65-40/' + method,
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study_withsir/0.65-70/' + method,
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study_withsir/0.65-90/' + method,
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study_withsir/0.65-110/' + method,
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/extra_withsir/0.25-110/MVAE_onehot',
            prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/0.160/' + method,
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/0.160/' + method,
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/0.160/' + method,
        ]
        # # rep_metrics_path = [
        # #     prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study/0.65-20/ILRMA/metrics-makemix_same_00.pkl',
        # #     prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study/0.65-30/ILRMA/metrics-makemix_same_00.pkl',
        # #     prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study/0.65-40/ILRMA/metrics-makemix_same_00.pkl',
        # #     prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study/0.65-70/ILRMA/metrics-makemix_same_00.pkl',
        # #     prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study/0.65-90/ILRMA/metrics-makemix_same_00.pkl',
        # #     prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study/0.65-110/ILRMA/metrics-makemix_same_00.pkl',
        # #     # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study/0.160/ILRMA/metrics-makemix_same_00.pkl',
        # #     # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study/0.360/ILRMA/metrics-makemix_same_00.pkl',
        # #     # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study/0.610/ILRMA/metrics-makemix_same_00.pkl',
        # # ]
        rep_metrics_path = None

        stft_len = 0.064
        stft_hop = 0.016
        vae_model_path = prefix + '/PROJECT/CVAE_training_data/standard_onehot/paper_model/v1/state_dict--epoch=2000.pt'
        # vae_model_path = prefix + '/PROJECT/MVAE_speakerencode_data/output/test_librispeech_ge2e_500sp_15min_01/model/state_dict--epoch=3000.pt'
        ge2e_model_path = prefix + '/PROJECT/GE2E_speaker_encoder_data/test_auglibrispeech--64_01/model/state_dict--sub_epoch=1400.pt'
        embedding_dim = 500
        device = torch.device(0)
        get_seped_utter(rirdata_path, pairinfo_path, out_path, suffix_list=suffix_list, sub_state=sub_state, method=method, prefix=prefix, rep_metrics_path=rep_metrics_path,
                        stft_len=stft_len, stft_hop=stft_hop, fs=16000,
                        vae_model_path=vae_model_path, spkr_model_path=ge2e_model_path,
                        device=device, embedding_dim=embedding_dim)


if __name__ == "__main__":
    path_prefix_other = '/home/user/zhaoyi.gu/mnt/g2'
    path_prefix_g2 = '/data/hdd0/zhaoyigu'
    main(state=4, sub_state=0, prefix=path_prefix_g2)

import librosa
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.realpath(os.path.dirname(__file__)))))
import torch
import pickle
from itertools import combinations, product
import random
import numpy as np
import librosa 
import scipy.io as sio
import soundfile as sf
from scipy.signal import convolve
import mir_eval
from Network.SepNet import vae
from Network.SpkrNet.ge2e import speaker_encoder_ge2e
from SepAlgo.ilrma import myilrma
from SepAlgo.mvae import mvae_onehot, mvae_ge2e
from utils import mysort, zero_pad, assign_rir


def make_mix2pair(srcinfo_path, outfile_path, random_mode=False, add_mode=False, get_n_pair=False,
                  n_pair=None, n_egs_per_pair=None, orifile_path=None, n_egs_all=None):
    """ 最新 """
    """ Make mixpairs """
    """ 
    Args:
        srcinfo_path: [str] path where test_signal_spk2utt.pkl is saved
        outfile_path: [str] path where mix2pair will be saved
        random_mode: [bool] if True, n_egs_all must be provided and this method randomly choose n_egs_all pair from all possible pairs from spk2utt;
                            if False, n_pair and n_egs_per_pair must be provided
        add_mode: [bool] if True, orifile_path where other mix2pair.pkl is saved must be provided, and new pair will not replicate those in orifile_path
        get_n_pair: [bool] if True, the method outputs total number of speaker pairs and return
        orifile_path: if provided, must be list containing all the other mix2pair.pkl
        n_pair, n_egs_per_pair: [int] if add_mode is True, means total value including those from orifile_path
        n_egs_all: [int] if add_mode is True, means total value including those from orifile_path
    
    Out:
        makemix: [dict]
                 [[(absolute-path-to-wav, offset, dur), (absolute-path-to-wav, offset, dur)], [(), ()], ...]
    """
    os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
    with open(srcinfo_path, 'rb') as f:
        spk2utt = pickle.load(f)
    spkr_list = list(spk2utt.keys())
    spkr_pair = list(combinations(spkr_list, 2))
    if get_n_pair:
        print("Total number of available pairs is {}".format(len(spkr_pair)))
        return
    mix2pair = []
    used_pair = {}
    compensate_cnt = 0
    num_exist_pair = 0

    if add_mode:
        # fill used_pair with already generated pairs
        for orifile in orifile_path:
            with open(orifile, 'rb') as f:
                mix2pair_ori = pickle.load(f)
            num_exist_pair += len(mix2pair)
            for (utt_id_1, path_1, _, _), (utt_id_2, path_2, _, _) in mix2pair_ori:
                spkr_name = [os.path.basename(path_1)[0: -4], os.path.basename(path_2)[0: -4]]
                utt_id = [utt_id_1, utt_id_2]
                index, spkr_name = mysort(spkr_name)
                if "{}-{}".format(spkr_name[0], spkr_name[1]) not in used_pair:
                    used_pair["{}-{}".format(spkr_name[0], spkr_name[1])] = []
                used_pair["{}-{}".format(spkr_name[0], spkr_name[1])].append((utt_id[index[0]], utt_id[index[1]]))

    if not random_mode:
        # i.e. Create pairs using mode "n_egs_per_pair egs from n_pair pairs"
        if len(spkr_pair) < n_pair:
            raise ValueError("n_pair exceeds the total number of available pairs--{}, you may need to reconsider n_pair and n_egs_per_pair".format(len(spkr_pair)))
        random.shuffle(spkr_pair)
        if use_all_speakers:
            spkr_set = defaultdict(int)
            num_generated_pair = 0
            spkr_pair_idx = 0
            while len(spkr_set.keys()) < len(spkr_list) or num_generated_pair < n_pair:
                if num_generated_pair < n_pair:
                    for spkr_name in spkr_pair[spkr_pair_idx]:
                        spkr_set[spkr_name] += 1


        for cnt, spkr_name in enumerate(spkr_pair[0: n_pair]):
            print(cnt)
            _, spkr_name = mysort(spkr_name)
            utt_pair = list(product(*[range(len(spk2utt[spkr_name[i]])) for i in range(len(spkr_name))]))
            utt_pair_num = len(utt_pair)
            if utt_pair_num < n_egs_per_pair:
                print("Total utterance combination for {} and {} is less than n_egs_per_pair, will be compensated using random mode".format(spkr_name[0], spkr_name[1]))
                compensate_cnt += n_egs_per_pair - utt_pair_num

            """ 需要根据声源数修改 """
            if "{}-{}-{}".format(spkr_name[0], spkr_name[1], spkr_name[2]) not in used_pair:
                used_pair["{}-{}-{}".format(spkr_name[0], spkr_name[1], spkr_name[2])] = []
                n_egs = min(len(utt_pair), n_egs_per_pair)
            else:
                for item in used_pair["{}-{}-{}".format(spkr_name[0], spkr_name[1], spkr_name[2])]:
                    utt_pair.remove(item)
                assert len(utt_pair) + len(used_pair["{}-{}-{}".format(spkr_name[0], spkr_name[1], spkr_name[2])]) == utt_pair_num
                n_egs = min(len(utt_pair), n_egs_per_pair - len(used_pair["{}-{}-{}".format(spkr_name[0], spkr_name[1], spkr_name[2])]))
                
            # random.shuffle(utt_pair)
            utt_select = random.sample(utt_pair, n_egs)
            # for utt_id_all in utt_pair[0: n_egs]:
            for utt_id_all in utt_select:
                used_pair["{}-{}-{}".format(spkr_name[0], spkr_name[1], spkr_name[2])].append(utt_id_all)
                mix2pair.append([spk2utt[spkr_name[i]][utt_id_all[i]] for i in range(len(spkr_name))])

    elif random_mode or (compensate_cnt > 0):
        # i.e. Create pairs using random mode
        all_egs = []
        for spkr1, spkr2 in spkr_pair:
            spkr_name = [spkr1, spkr2]
            _, spkr_name = mysort(spkr_name)
            utt_pair = list(product(range(len(spk2utt[spkr_name[0]])), range(len(spk2utt[spkr_name[1]]))))
            utt_pair_num = len(utt_pair)
            if "{}-{}".format(spkr_name[0], spkr_name[1]) in used_pair:
                for item in used_pair["{}-{}".format(spkr_name[0], spkr_name[1])]:
                    utt_pair.remove(item)
            assert len(utt_pair) + len(used_pair["{}-{}".format(spkr_name[0], spkr_name[1])]) == utt_pair_num
            all_egs.extend([(spkr_name[0], spkr_name[1], utt[0], utt[1]) for utt in utt_pair])
        random.shuffle(all_egs)
        if compensate_cnt > 0:
            n_egs = compensate_cnt
        else:
            n_egs = n_egs_all - num_exist_pair
        for spkr1, spkr2, utt_id_1, utt_id_2 in all_egs[0: n_egs]:
            mix2pair.append([spk2utt[spkr1][utt_id_1], spk2utt[spkr2][utt_id_2]])
    
    with open(outfile_path, 'wb') as f:
        pickle.dump(mix2pair, f)

        
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
    f_txt.close()

    spk2utt = {}    
    enroll_len = int(enroll_len * fs)
    utt_len = int(utt_len * fs)
    for src_path, dur in info:
        spkr_name = os.path.basename(src_path)[0: -4]
        if spkr_name not in spk2utt:
            spk2utt[spkr_name] = []
        num_utt = (dur - enroll_len) // utt_len
        for i in range(num_utt):
            spk2utt[spkr_name].append((i, src_path, enroll_len + i * utt_len, utt_len))
    
    with open(os.path.join(out_path, 'test_signal_spk2utt.pkl'), 'wb') as f:
        pickle.dump(spk2utt, f)


def make_mix2pair(srcinfo_path, outfile_path, random_mode=False, add_mode=False, get_n_pair=False,
                  n_pair=None, n_egs_per_pair=None, orifile_path=None, n_egs_all=None):
    """ Make mixpairs """
    """ 
    Args:
        srcinfo_path: [str] path where test_signal_spk2utt.pkl is saved
        outfile_path: [str] path where mix2pair will be saved
        random_mode: [bool] if True, n_egs_all must be provided and this method randomly choose n_egs_all pair from all possible pairs from spk2utt;
                            if False, n_pair and n_egs_per_pair must be provided
        add_mode: [bool] if True, orifile_path where other mix2pair.pkl is saved must be provided, and new pair will not replicate those in orifile_path
        get_n_pair: [bool] if True, the method outputs total number of speaker pairs and return
        orifile_path: if provided, must be list containing all the other mix2pair.pkl
        n_pair, n_egs_per_pair: [int] if add_mode is True, means total value including those from orifile_path
        n_egs_all: [int] if add_mode is True, means total value including those from orifile_path
    
    Out:
        makemix: [dict]
                 [[(absolute-path-to-wav, offset, dur), (absolute-path-to-wav, offset, dur)], [(), ()], ...]
    """
    os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
    with open(srcinfo_path, 'rb') as f:
        spk2utt = pickle.load(f)
    spkr_list = list(spk2utt.keys())
    spkr_pair = list(combinations(spkr_list, 2))
    if get_n_pair:
        print("Total number of available pairs is {}".format(len(spkr_pair)))
        return
    mix2pair = []
    used_pair = {}
    compensate_cnt = 0
    num_exist_pair = 0

    if add_mode:
        # fill used_pair with already generated pairs
        for orifile in orifile_path:
            with open(orifile, 'rb') as f:
                mix2pair_ori = pickle.load(f)
            num_exist_pair += len(mix2pair)
            for (utt_id_1, path_1, _, _), (utt_id_2, path_2, _, _) in mix2pair_ori:
                spkr_name = [os.path.basename(path_1)[0: -4], os.path.basename(path_2)[0: -4]]
                utt_id = [utt_id_1, utt_id_2]
                index, spkr_name = mysort(spkr_name)
                if "{}-{}".format(spkr_name[0], spkr_name[1]) not in used_pair:
                    used_pair["{}-{}".format(spkr_name[0], spkr_name[1])] = []
                used_pair["{}-{}".format(spkr_name[0], spkr_name[1])].append((utt_id[index[0]], utt_id[index[1]]))

    if not random_mode:
        # i.e. Create pairs using mode "n_egs_per_pair egs from n_pair pairs"
        if len(spkr_pair) < n_pair:
            raise ValueError("n_pair exceeds the total number of available pairs--{}, you may need to reconsider n_pair and n_egs_per_pair".format(len(spkr_pair)))
        random.shuffle(spkr_pair)
        for spkr1, spkr2 in spkr_pair[0: n_pair]:
            spkr_name = [spkr1, spkr2]
            _, spkr_name = mysort(spkr_name)
            utt_pair = list(product(range(len(spk2utt[spkr_name[0]])), range(len(spk2utt[spkr_name[1]]))))
            utt_pair_num = len(utt_pair)
            if utt_pair_num < n_egs_per_pair:
                print("Total utterance combination for {} and {} is less than n_egs_per_pair, will be compensated using random mode".format(spkr1, spkr2))
                compensate_cnt += n_egs_per_pair - utt_pair_num

            if "{}-{}".format(spkr_name[0], spkr_name[1]) not in used_pair:
                used_pair["{}-{}".format(spkr_name[0], spkr_name[1])] = []
                n_egs = min(len(utt_pair), n_egs_per_pair)
            else:
                for item in used_pair["{}-{}".format(spkr_name[0], spkr_name[1])]:
                    utt_pair.remove(item)
                assert len(utt_pair) + len(used_pair["{}-{}".format(spkr_name[0], spkr_name[1])]) == utt_pair_num
                n_egs = min(len(utt_pair), n_egs_per_pair - len(used_pair["{}-{}".format(spkr_name[0], spkr_name[1])]))
                
            random.shuffle(utt_pair)
            for utt_id_1, utt_id_2 in utt_pair[0: n_egs]:
                used_pair["{}-{}".format(spkr_name[0], spkr_name[1])].append((utt_id_1, utt_id_2))
                mix2pair.append([spk2utt[spkr_name[0]][utt_id_1], spk2utt[spkr_name[1]][utt_id_2]])

    elif random_mode or (compensate_cnt > 0):
        # i.e. Create pairs using random mode
        all_egs = []
        for spkr1, spkr2 in spkr_pair:
            spkr_name = [spkr1, spkr2]
            _, spkr_name = mysort(spkr_name)
            utt_pair = list(product(range(len(spk2utt[spkr_name[0]])), range(len(spk2utt[spkr_name[1]]))))
            utt_pair_num = len(utt_pair)
            if "{}-{}".format(spkr_name[0], spkr_name[1]) in used_pair:
                for item in used_pair["{}-{}".format(spkr_name[0], spkr_name[1])]:
                    utt_pair.remove(item)
            assert len(utt_pair) + len(used_pair["{}-{}".format(spkr_name[0], spkr_name[1])]) == utt_pair_num
            all_egs.extend([(spkr_name[0], spkr_name[1], utt[0], utt[1]) for utt in utt_pair])
        random.shuffle(all_egs)
        if compensate_cnt > 0:
            n_egs = compensate_cnt
        else:
            n_egs = n_egs_all - num_exist_pair
        for spkr1, spkr2, utt_id_1, utt_id_2 in all_egs[0: n_egs]:
            mix2pair.append([spk2utt[spkr1][utt_id_1], spk2utt[spkr2][utt_id_2]])
    
    with open(outfile_path, 'wb') as f:
        pickle.dump(mix2pair, f)
            

def get_seped_utter(rirdata_path, pairinfo_path, out_path, sub_state=0, method='ILRMA', prefix=None, **kwargs):
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

    f_idx2rir = open(os.path.join(rirdata_path, 'idx2rir.pkl'), 'rb')
    idx2rir = pickle.load(f_idx2rir)
    f_idx2rir.close()
    nfft = int(kwargs['fs'] * kwargs['stft_len']) 
    hop = int(kwargs['fs'] * kwargs['stft_hop'])
    fs = kwargs['fs']

    if method == 'MVAE_onehot':
        modelfile_path = kwargs['vae_model_path']
        n_embedding = kwargs['embedding_dim']
        device = kwargs['device']
        model = vae.net(n_embedding)
        model.load_state_dict(torch.load(modelfile_path, map_location=device))
        model.to(device)
        model.eval()
    elif method == 'MVAE_ge2e':
        fb_mat = torch.from_numpy(librosa.filters.mel(16000, nfft, n_mels=40)).unsqueeze(0)
        vaemodel_path = kwargs['vae_model_path']
        spkrmodel_path = kwargs['spkr_model_path']
        n_embedding = kwargs['embedding_dim']
        device = kwargs['device']
        vae_model = vae.net(n_embedding)
        vae_model.load_state_dict(torch.load(vaemodel_path, map_location=device))
        vae_model.to(device)
        vae_model.eval()
        spkr_model = speaker_encoder_ge2e()
        spkr_model_dict = spkr_model.state_dict()
        pretrained_dict = torch.load(spkrmodel_path, map_location=torch.device('cpu'))
        pretrained_dict_rename = {}
        for k, v in pretrained_dict.items():
            try:
                param_name = k.split('.', 1)[1]
                pretrained_dict_rename[param_name] = v
            except IndexError:
                pass
        pretrained_dict_rename = {k: v for k, v in pretrained_dict_rename.items() if k in spkr_model_dict}
        spkr_model_dict.update(pretrained_dict_rename)
        spkr_model.load_state_dict(spkr_model_dict)
        spkr_model.cuda(device)
        spkr_model.eval()
    
    for path_idx, pairinfo in enumerate(pairinfo_path):
        print('Current mix2pair is {}'.format(pairinfo))
        os.makedirs(out_path[path_idx], exist_ok=True)

        f_makemix = open(pairinfo, 'rb')
        makemix = pickle.load(f_makemix)
        f_makemix.close()
        if sub_state == 0:
            f_metrics = open(os.path.join(out_path[path_idx], 'metrics-{}.pkl'.format(os.path.basename(pairinfo[0: -4]))), 'wb')
        metric_dict = {'mix': [], 'SDR': {'ori': [], 'sep': []}, 'SIR': {'ori': [], 'sep': []}, 'SAR': {'ori': [], 'sep': []}, 'PERM': {'sep': []}}

        np.random.seed(0)
        torch.manual_seed(0)
        for idx, ((spkr_id_1, srcdata_path_1, offset_1, duration_1), (spkr_id_2, srcdata_path_2, offset_2, duration_2), gidx_rir) in enumerate(makemix):
            print("Processing {}/{} mixture.".format(idx + 1, len(makemix)))
            # Generate mixture and source signals
            src_name_1 = os.path.basename(srcdata_path_1)[0: -4]
            src_name_2 = os.path.basename(srcdata_path_2)[0: -4]
            out_path_necessary = os.path.join(out_path[path_idx], "{}-{}_{}-{}_{}".format(src_name_1, spkr_id_1, src_name_2, spkr_id_2, method))
            if prefix is not None:
                # srcdata_path_1 = os.path.join(prefix, srcdata_path_1.split('/', 4)[-1])
                # srcdata_path_2 = os.path.join(prefix, srcdata_path_2.split('/', 4)[-1])
                srcdata_path_1 = os.path.join(prefix, 'DATASET/Librispeech/concatenate/test_clean/' + src_name_1 + '.wav')
                srcdata_path_2 = os.path.join(prefix, 'DATASET/Librispeech/concatenate/test_clean/' + src_name_2 + '.wav')
            src_1, _ = sf.read(srcdata_path_1, start=offset_1, stop=offset_1 + duration_1)
            src_2, _ = sf.read(srcdata_path_2, start=offset_2, stop=offset_2 + duration_2)
            # src_1 = src_1 - np.mean(src_1)
            # src_2 = src_2 - np.mean(src_2)
            # src_1 = (src_1 - np.mean(src_1)) / np.max(np.abs(src_1 - np.mean(src_1))) + np.mean(src_1)
            src_1 = src_1 / np.max(np.abs(src_1))
            src_2 = src_2 * np.std(src_1) / np.std(src_2)
            assert src_1.shape[0] == src_2.shape[0]

            # rir = idx2rir[549][1]  # if the rir is simulated
            # mix_1 = convolve(src_1, rir[0][0]) + convolve(src_2, rir[0][1])
            # mix_2 = convolve(src_1, rir[1][0]) + convolve(src_2, rir[1][1])
            rir_path_1, rir_path_2 = idx2rir[gidx_rir]['path']
            channel = idx2rir[gidx_rir]['channel']
            rir_path_1 = os.path.join(prefix, rir_path_1)
            rir_path_2 = os.path.join(prefix, rir_path_2)
            rir_1 = sio.loadmat(rir_path_1)
            rir_2 = sio.loadmat(rir_path_2)
            rir_1 = rir_1['impulse_response'][:, channel].T
            rir_2 = rir_2['impulse_response'][:, channel].T
            rir_1 = librosa.core.resample(np.asfortranarray(rir_1), 48000, 16000)
            rir_2 = librosa.core.resample(np.asfortranarray(rir_2), 48000, 16000)
            rir_1 = rir_1[:, 0: int(0.8 * 16000)]
            rir_2 = rir_2[:, 0: int(0.8 * 16000)]
            mix_1 = convolve(src_1, rir_1[0, :]) + convolve(src_2, rir_2[0, :])
            mix_2 = convolve(src_1, rir_1[1, :]) + convolve(src_2, rir_2[1, :])

            mix = np.stack((mix_1, mix_2), axis=1)
            src = np.stack((src_1, src_2), axis=1)
            mix = mix[0: src.shape[0], :]
            if not method == "ILRMA":
                src = zero_pad(src.T, 4, hop_length=hop)
                mix = zero_pad(mix.T, 4, hop_length=hop)
                src, mix = src.T, mix.T

            # Separate mixture using method
            mix_spec = [librosa.core.stft(np.asfortranarray(mix[:, ch]), n_fft=nfft, hop_length=hop, win_length=nfft) for ch in range(mix.shape[1])]
            mix_spec = np.stack(mix_spec, axis=1)
            if method == 'ILRMA':
                sep_spec = myilrma(mix_spec, 1000, n_basis=2)
            elif method == "MVAE_onehot":
                sep_spec = mvae_onehot(mix_spec.swapaxes(1, 2), model, n_iter=1000, device=device)
            elif method == "MVAE_ge2e":
                sep_spec = mvae_ge2e(mix_spec.swapaxes(1, 2), vae_model, spkr_model, fb_mat=fb_mat, n_iter=1000, device=device)
            sep = [librosa.core.istft(sep_spec[:, ch, :], hop_length=hop, length=src.shape[0]) for ch in range(sep_spec.shape[1])]
            sep = np.stack(sep, axis=1)
            # sep_nm = copy.deepcopy(sep)
            # mix_nm = copy.deepcopy(mix)
            # src_nm = copy.deepcopy(src)
            # sep_nm = sep_nm - np.mean(sep_nm, axis=0, keepdims=True)
            # mix_nm = mix_nm - np.mean(mix_nm, axis=0, keepdims=True)
            # src_nm = src_nm - np.mean(src_nm, axis=0, keepdims=True)
            metrics = mir_eval.separation.bss_eval_sources(src.T, sep.T)
            metrics_ori = mir_eval.separation.bss_eval_sources(src.T, mix.T)

            metric_dict['mix'].append(makemix[idx])
            for i, m in enumerate(['SDR', 'SIR', 'SAR']):
                metric_dict[m]['sep'].extend(metrics[i].tolist())
                metric_dict[m]['ori'].extend(metrics_ori[i].tolist())
            metric_dict['PERM']['sep'].append(metrics[-1])
            
            if sub_state == 0:
                sf.write(out_path_necessary + '_sep.wav', sep, fs)
                with open(os.path.join(out_path[path_idx], 'log-{}.txt'.format(os.path.basename(pairinfo[0: -4]))), 'at') as f_log:
                    print('Finish processing index {} for {}'.format(idx, os.path.basename(pairinfo[0: -4])), file=f_log)
            if sub_state == 2:
                sf.write(out_path_necessary + '_mixtest.wav', mix, fs)
                sf.write(out_path_necessary + '_srctest.wav', src, fs)
                sf.write(out_path_necessary + '_septest.wav', sep, fs)

        if sub_state == 0:
            pickle.dump(metric_dict, f_metrics)
            f_metrics.close()
        

def main(state=0, sub_state=0, prefix=None):
    if state == 0:
        """ split concatenated test signal into small segments and obtain test_signal_spk2utt.pkl """
        data_path = prefix + '/DATASET/Librispeech/concatenate/test_clean'
        out_path = prefix + '/DATASET/Librispeech/concatenate/test_clean'
        enroll_len = 60
        utt_len = 5
        if sub_state == 0:
            make_src_signal(data_path, out_path, enroll_len=enroll_len, utt_len=utt_len, fs=16000)

    elif state == 1: 
        """ Create mixture pairs for testing, i.e. mix2pair.pkl """
        srcinfo_path = prefix + '/DATASET/Librispeech/concatenate/test_clean/test_signal_spk2utt.pkl'
        outfile_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study/mix2pair_same_00.pkl'
        random_mode = False
        add_mode = False
        n_pair = 330
        n_egs_per_pair = 1
        if sub_state == 0:
            make_mix2pair(srcinfo_path, outfile_path, random_mode=random_mode, add_mode=add_mode, n_pair=n_pair, n_egs_per_pair=n_egs_per_pair)
        elif sub_state == 1:
            """ Find total number of speaker pairs """
            make_mix2pair(srcinfo_path, outfile_path, get_n_pair=True)
        elif sub_state == 2:
            with open(outfile_path, 'rb') as f:
                mix2pair = pickle.load(f)
                dummy = 1

    elif state == 2:
        """ Assign rir to mix2pair """
        rirfile_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/RIR/test_real/key2idx.pkl'
        targetfile_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study/mix2pair_same_00.pkl'
        assign_key = "0.160"
        outfile_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study/' + assign_key + '/makemix_same_00.pkl'
        if sub_state == 0:
            assign_rir(rirfile_path, targetfile_path, outfile_path, assign_key=assign_key, random_choice=False)
        if sub_state == 1:
            """ Check keys """
            with open(rirfile_path, 'rb') as f:
                key2gidx = pickle.load(f)
            print(key2gidx.keys())

    elif state == 3:
        """ Using separation algorithms to get separated signals """
        # rirdata_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/RIR/test_clean/t60_angle_interval_study/'
        rirdata_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/RIR/test_real/'
        pairinfo_path = [
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean/t60_angle_interval_study/0.65-20/makemix_same_00.pkl',
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean/t60_angle_interval_study/0.65-30/makemix_same_00.pkl',
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean/t60_angle_interval_study/0.65-40/makemix_same_00.pkl',
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean/t60_angle_interval_study/0.65-70/makemix_same_00.pkl',
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean/t60_angle_interval_study/0.65-90/makemix_same_00.pkl',
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean/t60_angle_interval_study/0.65-110/makemix_same_00.pkl',
            prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study/0.360/makemix_same_00.pkl'
        ]
        out_path = [
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean/t60_angle_interval_study/0.65-20/MVAE_ge2e',
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean/t60_angle_interval_study/0.65-30/MVAE_ge2e',
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean/t60_angle_interval_study/0.65-40/MVAE_ge2e',
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean/t60_angle_interval_study/0.65-70/MVAE_ge2e',
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean/t60_angle_interval_study/0.65-90/MVAE_ge2e',
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean/t60_angle_interval_study/0.65-110/MVAE_ge2e',
            prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real/t60_study/0.360/MVAE_ge2e'
        ]

        method = "MVAE_ge2e"
        stft_len = 0.064
        stft_hop = 0.016
        # vae_model_path = prefix + '/PROJECT/MVAE_w_embds_data/output/onehot_embds/test_librispeech_500sp_15min/model/state_dict--epoch=1000.pt'
        vae_model_path = prefix + '/PROJECT/MVAE_speakerencode_data/output/test_librispeech_ge2e_500sp_15min_01/model/state_dict--epoch=3000.pt'
        ge2e_model_path = prefix + '/PROJECT/GE2E_speaker_encoder_data/test_auglibrispeech--64_01/model/state_dict--sub_epoch=1400.pt'
        embedding_dim = 64
        device = torch.device(0)
        get_seped_utter(rirdata_path, pairinfo_path, out_path, sub_state=sub_state, method=method, prefix=prefix,
                        stft_len=stft_len, stft_hop=stft_hop, fs=16000,
                        vae_model_path=vae_model_path, spkr_model_path=ge2e_model_path,
                        device=device, embedding_dim=embedding_dim)
        

if __name__ == "__main__":
    path_prefix_other = '/home/user/zhaoyi.gu/mnt/g2'
    path_prefix_g2 = '/data/hdd0/zhaoyigu'
    main(state=3, sub_state=0, prefix=path_prefix_g2)

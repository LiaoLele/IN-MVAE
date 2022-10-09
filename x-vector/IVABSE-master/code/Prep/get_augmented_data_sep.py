import librosa
import os
import pickle
import random
import soundfile as sf
from scipy.signal import convolve
import numpy as np
import mir_eval
from Sep_Algo.ilrma import myilrma
import copy
from utils import assign_rir, get_uttnum


def split_concatenate_data(data_path, out_path, uttlen_limit, fs=16000):
    """ Split concatenated data into utterances[not creating real data, but using starting point and duration to represent] """
    """ Make utterance pairs for separation """
    """
    Args:
        `data_path`: [str] path where concatenated dataset is saved and spk2utt.pkl will be saved 
        `out_path`: [str] path where mix2pair.pkl will be saved
        `uttlen_limit`: [tuple/list] upper and lower bound of utterance length [in seconds]
        `fs`: [int] sampling rate
    
    Out:
        `spk2utt`: [Not returned but saved][dict]
                   {‘spkr-id-0’: [(datapath, start-0, dur-0), (datapath, start-1, dur-1), …],
                    ‘spkr-id-1’: [(), (), ...], ...}
        `mix2pair`: [Not returned but saved][list]
                    [[(spkr-id-0-a, datapath, start-0-a, dur-0-a), (spkr-id-0-b, datapath, start-0-b, dur-0-b)], [(), ()], ...]
        `spk2utt.pkl`: pickle file that saves spk2utt
        `mix2pair.pkl`: pickle file that saves mix2pair
    """
    os.makedirs(out_path, exist_ok=True)

    f = open(os.path.join(data_path, 'info.pkl'), 'rb')
    data_info = pickle.load(f)
    f.close()

    f_spk2utt = open(os.path.join(data_path, 'spk2utt.pkl'), 'wb')
    f_mix2pair = open(os.path.join(out_path, 'mix2pair.pkl'), 'wb')
    spk2utt = {}
    mix2pair = []
    
    def spk2utt_add(spk2utt, spkr_id):
        """ add new speaker key to spk2utt """
        if spkr_id not in spk2utt:
            spk2utt[spkr_id] = []

    # Initialization
    uttlen_limit = [int(x * fs) for x in uttlen_limit]
    data_info = sorted(data_info, key=lambda x: x[1], reverse=True)
    spkr_num = len(data_info)
    used_len = [int(0)] * spkr_num
    cnt = 0

    for i, (spkr_path, spkr_len, spkr_id) in enumerate(data_info):
        spk2utt_add(spk2utt, spkr_id)
        while True:
            if (spkr_len - used_len[spkr_id]) < uttlen_limit[0]:
                print('Finish processing {}'.format(os.path.basename(spkr_path)))
                break
            uttlen = random.randint(uttlen_limit[0], min(uttlen_limit[1], spkr_len - used_len[spkr_id])) 
            spk2utt[spkr_id].append((spkr_path, used_len[spkr_id], uttlen))

            rest_spkr_list = list(range(i + 1, spkr_num))
            random.shuffle(rest_spkr_list)
            find_pair = False
            for j in rest_spkr_list:
                spkr_path_2, spkr_len_2, spkr_id_2 = data_info[j]
                assert not spkr_id == spkr_id_2
                if (spkr_len_2 - used_len[spkr_id_2]) < uttlen:
                    continue
                else:
                    spk2utt_add(spk2utt, spkr_id_2)
                    spk2utt[spkr_id_2].append((spkr_path_2, used_len[spkr_id_2], uttlen))
                    mix2pair.append([(spkr_id, spkr_path, used_len[spkr_id], uttlen), (spkr_id_2, spkr_path_2, used_len[spkr_id_2], uttlen)])
                    used_len[spkr_id_2] += uttlen
                    find_pair = True
                    break
            if not find_pair:
                cnt += 1
                spkr_path_2, spkr_len_2, spkr_id_2 = data_info[random.choice(list(range(0, i)) + list(range(i + 1, spkr_num)))]
                assert not spkr_id == spkr_id_2
                mix2pair.append([(spkr_id, spkr_path, used_len[spkr_id], uttlen), (spkr_id_2, spkr_path_2, random.randint(0, spkr_len_2 - uttlen), uttlen)])
            used_len[spkr_id] += uttlen

    print('{} utterances in mix2pair might concern overlap.'.format(cnt))
    pickle.dump(spk2utt, f_spk2utt)
    pickle.dump(mix2pair, f_mix2pair)
    f_spk2utt.close()
    f_mix2pair.close()


def split_makemix(utt_path, ncpu=1):
    """ Split makemix.pkl into multiple orthognal files i.e. makemix.1.pkl, makemix.2.pkl ... """
    """ so that separation can be performed using many cpus """
    """
    Args:
        `utt_path`: [str] path where makemix.pkl is saved
        `ncpu`: [int] number to split

    Out:
        `makemix.n.pkl`: pickle file that saves a part of makemix.pkl
    """
    f_makemix = open(os.path.join(utt_path, 'makemix.pkl'), 'rb')
    makemix = pickle.load(f_makemix)
    f_makemix.close()

    mix_num = len(makemix)
    mix_num_per_cpu = [mix_num // ncpu] * (ncpu - 1)
    mix_num_per_cpu.append(mix_num - mix_num // ncpu * (ncpu - 1))
    assert sum(mix_num_per_cpu) == mix_num
    print('Use {} cpus, number of each cpu is {}'.format(ncpu, mix_num_per_cpu))

    start_idx = 0
    for i, local_mix_num in enumerate(mix_num_per_cpu):
        f_makemix_local = open(os.path.join(utt_path, 'makemix.{}.pkl'.format(i)), 'wb')
        pickle.dump(makemix[start_idx: start_idx + local_mix_num], f_makemix_local)
        f_makemix_local.close()
        start_idx += local_mix_num
    assert start_idx == mix_num


def get_seped_utter(rirdata_path, pairinfo_path, out_path, sub_state=0, job=0, method='ilrma', prefix=None, **kwargs):
    """ separate mixtures in makemix.n.pkl and save separated utterances to out_path """
    """
    Args:
        `rirdata_path`: [str] path where idx2rir.pkl is saved
        `pairinfo_path`: [str] path where makemix.n.pkl is saved
        `out_path`: [str] path where spk2utt.pkl will be saved
        `sub_state`: [int] 0 is execution state; 1 is debugging state
        `job`: [int] index of makemix.n.pkl
        `method`: [str] separation method
        `kwargs`: [dict] params for STFT and method

    Out:
        `spk2utt`: [Not returned but saved][dict]
                   dict object that saves spk2utt information of augmented (i.e. separated) dataset
                   {‘spkr-id-0’: [(datapath, start-0, dur-0), (datapath, start-1, dur-1), …],
                    ‘spkr-id-1’: [(), (), ...], ...}
        `metric_dict`: [Not returned but saved][dict]
                    dict object that saves SDR, SIR, SAR information of separated data in makemix.n.pkl order
                    {'mix': [makemix[0], makemix[1], ...], 'SDR': {'ori': [], 'sep': []}, 'SIR': {'ori': [], 'sep': []}, 'SAR': {'ori': [], 'sep': []}}
        `spk2utt.n.pkl`: pickle file that saves spk2utt for job n
        `metrics.n.pkl`: pickle file that saves metric_dict for job n
        `log.n.txt`: txt file that saves log for job n
                     each line of log is 'Finish processing index m for job n' 
    """
    print('Current job is {}'.format(job))
    os.makedirs(out_path, exist_ok=True)

    f_idx2rir = open(os.path.join(rirdata_path, 'idx2rir.pkl'), 'rb')
    f_makemix = open(os.path.join(pairinfo_path, 'makemix.{}.pkl'.format(job)), 'rb')
    idx2rir = pickle.load(f_idx2rir)
    makemix = pickle.load(f_makemix)
    f_idx2rir.close()
    f_makemix.close()
    
    nfft = int(kwargs['fs'] * kwargs['stft_len']) 
    hop = int(kwargs['fs'] * kwargs['stft_hop'])
    fs = kwargs['fs']
    if sub_state == 0:
        f_spk2utt = open(os.path.join(out_path, 'spk2utt.{}.pkl'.format(job)), 'wb')
        f_metrics = open(os.path.join(out_path, 'metrics.{}.pkl'.format(job)), 'wb')
    spk2utt = {}
    metric_dict = {'mix': [], 'SDR': {'ori': [], 'sep': []}, 'SIR': {'ori': [], 'sep': []}, 'SAR': {'ori': [], 'sep': []}}

    np.random.seed(job)
    for idx, ((spkr_id_1, srcdata_path_1, offset_1, duration_1), (spkr_id_2, srcdata_path_2, offset_2, duration_2), gidx_rir) in enumerate(makemix):
        # if idx <= 2559:
        #     continue
        print("Processing {}/{} mixture.".format(idx + 1, len(makemix)))
        if spkr_id_1 not in spk2utt:
            spk2utt[spkr_id_1] = []
        if spkr_id_2 not in spk2utt:
            spk2utt[spkr_id_2] = []
        # Generate mixture and source signals
        src_name_1 = os.path.basename(srcdata_path_1)[0: -4]
        src_name_2 = os.path.basename(srcdata_path_2)[0: -4]
        if prefix is not None:
            # srcdata_path_1 = os.path.join(prefix, srcdata_path_1.split('/', 4)[-1])
            # srcdata_path_2 = os.path.join(prefix, srcdata_path_2.split('/', 4)[-1])
            srcdata_path_1 = os.path.join(prefix, 'DATASET/Librispeech/concatenate_MN/train_clean/' + src_name_1 + '.wav')
            srcdata_path_2 = os.path.join(prefix, 'DATASET/Librispeech/concatenate_MN/train_clean/' + src_name_2 + '.wav')
        out_path_whole_1 = os.path.join(out_path, src_name_1)
        out_path_whole_2 = os.path.join(out_path, src_name_2)
        if not os.path.exists(out_path_whole_1):
            os.makedirs(out_path_whole_1, exist_ok=True)
        if not os.path.exists(out_path_whole_2):
            os.makedirs(out_path_whole_2, exist_ok=True)
        src_1, _ = sf.read(srcdata_path_1, start=offset_1, stop=offset_1 + duration_1)
        src_2, _ = sf.read(srcdata_path_2, start=offset_2, stop=offset_2 + duration_2)
        src_1 = src_1 - np.mean(src_1)
        src_2 = src_2 - np.mean(src_2)
        # src_1 = (src_1 - np.mean(src_1)) / np.max(np.abs(src_1 - np.mean(src_1))) + np.mean(src_1)
        src_1 = src_1 / np.max(np.abs(src_1))
        src_2 = src_2 * np.std(src_1) / np.std(src_2)
        assert src_1.shape[0] == src_2.shape[0]
        rir = idx2rir[gidx_rir][1]
        mix_1 = convolve(src_1, rir[0][0]) + convolve(src_2, rir[0][1])
        mix_2 = convolve(src_1, rir[1][0]) + convolve(src_2, rir[1][1])
        mix = np.stack((mix_1, mix_2), axis=1)
        src = np.stack((src_1, src_2), axis=1)
        mix = mix[0: src.shape[0], :]

        # Separate mixture using method
        mix_spec = [librosa.core.stft(np.asfortranarray(mix[:, ch]), n_fft=nfft, hop_length=hop, win_length=nfft) for ch in range(mix.shape[1])]
        mix_spec = np.stack(mix_spec, axis=1)
        if method == 'ilrma':
            sep_spec = myilrma(mix_spec, 1000, n_basis=2)
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
        
        if sub_state == 0:
            # TODO: 用pesq的结果可以筛选得更准确
            if metrics[-1][0] == 0:
                sf.write(os.path.join(out_path_whole_1, '{}-{}-{}-0-sep.wav'.format(job, idx, gidx_rir)), sep[:, 0], fs)
                sf.write(os.path.join(out_path_whole_2, '{}-{}-{}-1-sep.wav'.format(job, idx, gidx_rir)), sep[:, 1], fs)
            else:
                sf.write(os.path.join(out_path_whole_1, '{}-{}-{}-0-sep.wav'.format(job, idx, gidx_rir)), sep[:, 1], fs)
                sf.write(os.path.join(out_path_whole_2, '{}-{}-{}-1-sep.wav'.format(job, idx, gidx_rir)), sep[:, 0], fs)
            spk2utt[spkr_id_1].append((os.path.join(out_path_whole_1, '{}-{}-{}-0-sep.wav'.format(job, idx, gidx_rir)), 0, sep.shape[0]))
            spk2utt[spkr_id_2].append((os.path.join(out_path_whole_2, '{}-{}-{}-1-sep.wav'.format(job, idx, gidx_rir)), 0, sep.shape[0]))
            with open(os.path.join(out_path, 'log.{}.txt'.format(job)), 'at') as f_log:
                print('Finish processing index {} for job {}'.format(idx, job), file=f_log)

        if sub_state == 2:
            sf.write(os.path.join(out_path, '{}-{}-{}-mix.wav'.format(job, idx, gidx_rir)), mix, fs)
            sf.write(os.path.join(out_path, '{}-{}-src.wav'.format(job, idx)), src, fs)
            sf.write(os.path.join(out_path, '{}-{}-sep.wav'.format(job, idx)), sep, fs)

    if sub_state == 0:
        pickle.dump(spk2utt, f_spk2utt)
        pickle.dump(metric_dict, f_metrics)
        f_spk2utt.close()
        f_metrics.close()


def split_data_to_list(split_data_path, out_path, sub_state=0):
    """ Make spk2utt[dict] to list """
    """ 
    Args:
        `split_data_path`: path where spk2utt.pkl is saved
        `out_path`: path where split_data_list will be saved
    
    Out:
        `split_data_list`: [Not returned but saved][list]
                           [(spkr-id-0, spkr-path-0, spkr-offset-0, spkr-dur-0), (), ...]
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(split_data_path, 'rb') as f:
        spk2utt = pickle.load(f)
    numutt = get_uttnum(spk2utt)
    split_data_list = []
    for spkr_id in spk2utt.keys():
        for item in spk2utt[spkr_id]:
            item = list(item)
            item.insert(0, spkr_id)
            item = tuple(item)
            split_data_list.append([item])
    assert len(split_data_list) == numutt
    if sub_state == 0:
        with open(out_path, 'wb') as f:
            pickle.dump(split_data_list, f)


def get_reverb_utter(rir_data_path, datafile_path, out_path, sub_state=0, prefix=None):
    os.makedirs(out_path, exist_ok=True)
    with open(rir_data_path, 'rb') as f:
        idx2rir = pickle.load(f)
    with open(datafile_path, 'rb') as f:
        data = pickle.load(f)
    if sub_state == 0:
        f_log = open(os.path.join(out_path, 'log.txt'), 'at')
    spk2utt = {}
    for idx, ((spkr_id, srcdata_path, offset, duration), gidx_rir) in enumerate(data):
        print("Processing {}/{} utterance.".format(idx + 1, len(data)))
        if spkr_id not in spk2utt:
            spk2utt[spkr_id] = []
        src_name = os.path.basename(srcdata_path)[0: -4]
        if prefix is not None:
            srcdata_path = os.path.join(prefix, 'DATASET/Librispeech/concatenate_MN/train_clean/' + src_name + '.wav')
        out_path_whole = os.path.join(out_path, src_name)
        os.makedirs(out_path_whole, exist_ok=True)
        src, fs = sf.read(srcdata_path, start=offset, stop=offset + duration)
        src = src - np.mean(src)
        src = src / np.max(np.abs(src))
        rir = idx2rir[gidx_rir][1]
        rir_idx = random.choice([0, 1])
        out = convolve(src, rir[0][rir_idx])
        if sub_state == 0:
            sf.write(os.path.join(out_path_whole, '{}-{}.wav'.format(idx, gidx_rir)), out, fs)
            spk2utt[spkr_id].append((os.path.join(out_path_whole, '{}-{}.wav'.format(idx, gidx_rir)), 0, out.shape[0]))
            print('Finish processing index {}'.format(idx), file=f_log)
    if sub_state == 0:
        with open(os.path.join(out_path, 'spk2utt.pkl'), 'wb') as f:
            pickle.dump(spk2utt, f)


def get_spk2utt_from_interupt_sepjob(makemix_path, sepdata_path, job=None):
    print(job)
    with open(os.path.join(makemix_path, 'makemix.{}.pkl'.format(job)), 'rb') as f:
        makemix = pickle.load(f)
    spk2utt = {}
    for idx, ((spkr_id_1, srcdata_path_1, offset_1, duration_1), (spkr_id_2, srcdata_path_2, offset_2, duration_2), gidx_rir) in enumerate(makemix):
        src_name_1 = os.path.basename(srcdata_path_1)[0: -4]
        src_name_2 = os.path.basename(srcdata_path_2)[0: -4]
        if spkr_id_1 not in spk2utt:
            spk2utt[spkr_id_1] = []
        if spkr_id_2 not in spk2utt:
            spk2utt[spkr_id_2] = []

        sep_1 = list(filter(lambda x: x.startswith('{}-{}-{}-'.format(job, idx, gidx_rir)), os.listdir(os.path.join(sepdata_path, src_name_1))))
        sep_2 = list(filter(lambda x: x.startswith('{}-{}-{}-'.format(job, idx, gidx_rir)), os.listdir(os.path.join(sepdata_path, src_name_2))))
        assert len(sep_1) == 1
        assert len(sep_2) == 1
        sep_1 = os.path.join(os.path.join(sepdata_path, src_name_1, sep_1[0]))
        sep_2 = os.path.join(os.path.join(sepdata_path, src_name_2, sep_2[0]))
        sepdata_1, fs = sf.read(sep_1)
        sepdata_2, fs = sf.read(sep_2)
        spk2utt[spkr_id_1].append((sep_1, 0, sepdata_1.shape[0]))
        spk2utt[spkr_id_2].append((sep_2, 0, sepdata_2.shape[0]))
    
    with open(os.path.join(sepdata_path, 'spk2utt.{}.pkl'.format(job)), 'wb') as f:
        pickle.dump(spk2utt, f)
    

def main(state=0, sub_state=0, *, current_job=None, path_prefix='/data/hdd0/zhaoyigu'):

    if state == 0:
        """ Generate spk2utt.pkl and mix2pair.pkl """
        data_path = path_prefix + '/DATASET/Librispeech/concatenate/train_clean'
        out_path = path_prefix + '/DATASET/Librispeech/concatenate/train_clean'
        uttlen_limit = [5, 30]
        if sub_state == 0:
            split_concatenate_data(data_path, out_path, uttlen_limit, fs=16000)
        elif sub_state == 1:
            f_spk2utt = open(os.path.join(data_path, 'spk2utt.pkl'), 'rb')
            f_mix2pair = open(os.path.join(out_path, 'mix2pair.pkl'), 'rb')
            spk2utt = pickle.load(f_spk2utt)
            mix2pair = pickle.load(f_mix2pair)
            f_spk2utt.close()
            f_mix2pair.close()

    elif state == 1:
        """ assign rir """
        rir_path = path_prefix + '/DATASET/SEPARATED_LIBRISPEECH/RIR/train_clean/t602gidx.pkl'
        utt_path = path_prefix + '/DATASET/Librispeech/concatenate/train_clean/spk2utt_list.pkl'
        out_path = path_prefix + '/DATASET/SEPARATED_LIBRISPEECH/RIR/train_clean/makemix.pkl'
        if sub_state == 0:
            assign_rir(rir_path, utt_path, out_path)
        elif sub_state == 1:
            f = open(out_path, 'rb')
            makemix = pickle.load(f)
            f.close()

    elif state == 2:
        """ Split makemix.pkl into makemix.n.pkl """
        utt_path = path_prefix + '/DATASET/SEPARATED_LIBRISPEECH/RIR/train_clean'
        if sub_state == 0:
            split_makemix(utt_path, ncpu=12)
        elif sub_state == 1:
            f2 = open(os.path.join(utt_path, 'makemix.pkl'), 'wb')
            makemix = []
            for i in range(12):
                f = open(os.path.join(utt_path, 'makemix.{}.pkl'.format(i)), 'rb')
                makemix_tmp = pickle.load(f)
                f.close()
                makemix.extend(makemix_tmp)
            pickle.dump(makemix, f2)
            f2.close()

    elif state == 3:
        """ Create separated dataset """
        rirdata_path = path_prefix + '/DATASET/SEPARATED_LIBRISPEECH/RIR/train_clean'
        pairinfo_path = path_prefix + '/DATASET/SEPARATED_LIBRISPEECH/RIR/train_clean'
        out_path = path_prefix + '/DATASET/SEPARATED_LIBRISPEECH/DATA_MN/train_clean'
        stft_len = 0.064
        stft_hop = 0.016
        if sub_state == 1:
            f_test = open(os.path.join(out_path, 'spk2utt.1.pkl'), 'rb')
            print(os.path.join(out_path, 'spk2utt.1.pkl'))
            spk2utt = pickle.load(f_test)
            f_test.close()
        else:
            get_seped_utter(rirdata_path, pairinfo_path, out_path, sub_state=sub_state,
                            job=current_job, method='ilrma', prefix=path_prefix, stft_len=stft_len, stft_hop=stft_hop, fs=16000)

    elif state == 4:
        """ organize spk2utt from interuptted sepjob """
        makemix_path = path_prefix + '/DATASET/SEPARATED_LIBRISPEECH/RIR/train_clean'
        sepdata_path = path_prefix + '/DATASET/SEPARATED_LIBRISPEECH/DATA_MN/train_clean'
        if sub_state == 0:
            get_spk2utt_from_interupt_sepjob(makemix_path, sepdata_path, job=current_job)
        else:
            sepdata_path_ori = path_prefix + '/DATASET/SEPARATED_LIBRISPEECH/DATA/train_clean'
            with open(os.path.join(sepdata_path, 'spk2utt.7.pkl'), 'rb') as f:
                spk2utt_new = pickle.load(f)
            with open(os.path.join(sepdata_path_ori, 'spk2utt.7.pkl'), 'rb') as f:
                spk2utt_ori = pickle.load(f)
            for spk in spk2utt_new.keys():
                assert len(spk2utt_new[spk]) == len(spk2utt_ori[spk])
    
    elif state == 5:
        """ spk2utt to list """
        split_data_path = path_prefix + '/DATASET/Librispeech/concatenate/train_clean/spk2utt.pkl'
        out_path = path_prefix + '/DATASET/Librispeech/concatenate/train_clean/spk2utt_list.pkl'
        split_data_to_list(split_data_path, out_path, sub_state=sub_state)

    elif state == 6:
        """ Generate reverberated uttrance """
        rir_data_path = path_prefix + '/DATASET/SEPARATED_LIBRISPEECH/RIR/train_clean/idx2rir.pkl'
        datafile_path = path_prefix + '/DATASET/SEPARATED_LIBRISPEECH/RIR/train_clean/makesingle.pkl'
        out_path = '/home/user/zhaoyi.gu/DATA/reverb'
        get_reverb_utter(rir_data_path, datafile_path, out_path, sub_state=sub_state, prefix=path_prefix)
    

if __name__ == "__main__":
    path_prefix_other = '/home/user/zhaoyi.gu/mnt/g2'
    path_prefix_g2 = '/data/hdd0/zhaoyigu'
    main(state=1, sub_state=1, current_job=8, path_prefix=path_prefix_other)


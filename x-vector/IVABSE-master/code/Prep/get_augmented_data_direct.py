from collections import defaultdict
from scipy.signal import convolve
from utils import change_path
import soundfile as sf
import numpy as np
import argparse
import datetime
import librosa
import pickle
import random
import shlex
import os


def split_concatenate_data(data_path, out_path, uttlen_limit, fs=16000, seed=None, suffix=''):
    """ Split concatenated data into utterances[not creating real data, but using starting point and duration to represent] """
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
        `spk2utt.pkl`: pickle file that saves spk2utt
    """
    os.makedirs(out_path, exist_ok=True)
    random.seed(seed)

    f = open(os.path.join(data_path, 'info.pkl'), 'rb')
    data_info = pickle.load(f)
    f.close()

    spk2utt = {}
    utt = []
    
    def spk2utt_add(spk2utt, spkr_id):
        """ add new speaker key to spk2utt """
        if spkr_id not in spk2utt:
            spk2utt[spkr_id] = []

    # Initialization
    cnt = 0
    uttlen_limit = [int(x * fs) for x in uttlen_limit]
    data_info = sorted(data_info, key=lambda x: x[1], reverse=True)
    spkr_num = len(data_info)
    used_len = [int(0)] * spkr_num

    for i, (spkr_path, spkr_len, spkr_id) in enumerate(data_info):
        spk2utt_add(spk2utt, spkr_id)
        """ Need checking """
        spkr_path = spkr_path.split('/', 4)[-1]
        while True:
            if (spkr_len - used_len[spkr_id]) < uttlen_limit[0]:
                print('Finish processing {}'.format(os.path.basename(spkr_path)))
                break
            uttlen = random.randint(uttlen_limit[0], min(uttlen_limit[1], spkr_len - used_len[spkr_id])) 
            spk2utt[spkr_id].append((spkr_path, used_len[spkr_id], uttlen))
            utt.append((spkr_id, spkr_path, used_len[spkr_id], uttlen))
            used_len[spkr_id] += uttlen
            cnt += 1
    
    assert cnt == len(utt)
    print("In total, {} utterances with duration randomly chosen from {} s and {} s are generated!".format(cnt, int(uttlen_limit[0] / fs), int(uttlen_limit[1] / fs)))
    f_spk2utt = open(os.path.join(out_path, 'spk2utt_clean.' + suffix + '.pkl'), 'wb')
    pickle.dump(spk2utt, f_spk2utt)
    f_spk2utt.close()
    f_utt = open(os.path.join(out_path, 'cleanutt.' + suffix + '.pkl'), 'wb')
    pickle.dump(utt, f_utt)
    f_utt.close()


def parse_rir_list(rir_set_list_path):
    """ This function creates the RIR list
        Each rir object in the list contains the following attributes:
        rir_id, room_id, receiver_position_id, source_position_id, rt60, drr, probability
        Please refer to the help messages in the parser for the meaning of these attributes
    """
    rir_parser = argparse.ArgumentParser()
    rir_parser.add_argument('--rir-id', type=str, required=True, help='This id is unique for each RIR and the noise may associate with a particular RIR by refering to this id')
    rir_parser.add_argument('--room-id', type=str, required=True, help='This is the room that where the RIR is generated')
    rir_parser.add_argument('--receiver-position-id', type=str, default=None, help='receiver position id')
    rir_parser.add_argument('--source-position-id', type=str, default=None, help='source position id')
    rir_parser.add_argument('--rt60', type=float, default=None, help='RT60 is the time required for reflections of a direct sound to decay 60 dB.')
    rir_parser.add_argument('--drr', type=float, default=None, help='Direct-to-reverberant-ratio of the impulse response.')
    rir_parser.add_argument('--cte', type=float, default=None, help='Early-to-late index of the impulse response.')
    rir_parser.add_argument('--probability', type=float, default=None, help='probability of the impulse response.')
    rir_parser.add_argument('rir_rspecifier', type=str, help="""rir rspecifier, it can be either a filename or a piped command.
                            E.g. data/impulses/Room001-00001.wav or "sox data/impulses/Room001-00001.wav -t wav - |" """)

    rir_list = []
    for rir_set in rir_set_list_path:
        current_rir_list = [rir_parser.parse_args(shlex.split(x.strip())) for x in open(rir_set)]
        for rir in current_rir_list:
            rir_list.append(rir.rir_rspecifier)
    return rir_list


def assign_rir(oriutt_path, rir_set_list_path, out_path, suffix='', seed=None):
    rir_list = parse_rir_list(rir_set_list_path)
    random.seed(seed)
    random.shuffle(rir_list)

    with open(oriutt_path, 'rb') as f:
        oriutt_list = pickle.load(f)
    suffix_acc = oriutt_path.split('.')[1]
    suffix = suffix_acc + '-' + suffix
    revutt = []
    f_txt = open(os.path.join(out_path, 'revutt.' + suffix + '.txt'), 'wt')
    print("The total number of rir is {}, and of equal probability.".format(len(rir_list)), end='\n', file=f_txt)

    for utt in oriutt_list:
        speaker_rir = random.choice(rir_list)
        revutt.append((utt, speaker_rir))
        print(revutt[-1], file=f_txt)
    
    assert len(revutt) == len(oriutt_list)
    f_txt.close()
    with open(os.path.join(out_path, 'revutt.' + suffix + '.pkl'), 'wb') as f:
        pickle.dump(revutt, f)


def generate_rev_data(revutt_path, rir_path, out_path, path_prefix_hdd0='', path_prefix_ssd1='', sr=16000):
    """ Generated reverberated data according to revutt.pkl """
    with open(os.path.join(path_prefix_hdd0, revutt_path), 'rb') as f:
        revutt_list = pickle.load(f)
    suffix = revutt_path.split('.')[1]
    spk2utt = defaultdict(list)
    spk2uttidx = defaultdict(int)

    for idx, ((spkr_id, utt_path, start, dur), rel_rir_path) in enumerate(revutt_list):
        speaker_name = utt_path.rsplit('/', 1)[1].split('.')[0]
        out_dir = os.path.join(path_prefix_ssd1, out_path, speaker_name)
        os.makedirs(out_dir, exist_ok=True)
        out_name = 'rev_{}-{}.'.format(speaker_name, spk2uttidx[spkr_id]) + suffix + '.wav'
        spk2uttidx[spkr_id] += 1

        abs_rir_path = os.path.join(path_prefix_hdd0, rir_path, rel_rir_path)
        rir_sig, fs = sf.read(abs_rir_path)
        if fs != sr:
            rir_sig = librosa.resample(rir_sig, fs, sr)
        if len(rir_sig) > sr:
            print("Warning: rir length is longer than 1 second!")
        
        abs_src_path = os.path.join(path_prefix_hdd0, utt_path)
        src_sig, fs = sf.read(abs_src_path, start=start, stop=start + dur)
        if fs != sr:
            src_sig = librosa.resample(src_sig, fs, sr)
        src_sig = src_sig - np.mean(src_sig)
        src_std = np.std(src_sig)
        
        rev_sig = convolve(src_sig, rir_sig, mode='same')
        rev_sig = rev_sig * src_std / np.std(rev_sig)
        sf.write(os.path.join(out_dir, out_name), rev_sig, sr)

        spk2utt[spkr_id].append((os.path.join(out_path, speaker_name, out_name), 0, rev_sig.shape[-1]))
        print(idx)
    
    cnt = 0
    for utt in spk2utt.values():
        cnt += len(utt)
    assert cnt == len(revutt_list)
    assert len(spk2utt.keys()) == 1172
    
    with open(os.path.join(path_prefix_hdd0, os.path.dirname(revutt_path), 'spk2utt_rev.' + suffix + '.pkl'), 'wb') as f:
        pickle.dump(spk2utt, f)
    

def assign_bubble_noise(oriutt_path, bubble_path, out_path, snr_list=None, bubble_num_list=None,
                        suffix='', seed=None):
    with open(os.path.join(path_prefix_hdd0, oriutt_path), 'rb') as f:
        oriutt_list = pickle.load(f)
    with open(os.path.join(path_prefix_hdd0, bubble_path), 'rb') as f:
        spk2utt = pickle.load(f)
    suffix_acc = oriutt_path.split('.')[1]
    suffix = suffix_acc + '-' + suffix
    f_txt = open(os.path.join(out_path, 'bubbleutt.' + suffix + '.txt'), 'wt')
    
    random.seed(seed)
    bubbleutt = []
    for utt in oriutt_list:
        noises = []
        src_id = utt[0]
        bubble_id_list = set(spk2utt.keys())
        assert len(bubble_id_list) == 1172
        bubble_id_list.remove(src_id)
        bubble_id_list = list(bubble_id_list)
        bubble_num = random.choice(bubble_num_list)
        snrs = random.choices(snr_list, k=bubble_num)
        bubble_ids = random.choices(bubble_id_list, k=bubble_num)
        for bubble_id in bubble_ids:
            assert bubble_id != src_id
            noises.append(random.choice(spk2utt[bubble_id]))
        bubbleutt.append((utt, noises, snrs))
        print(bubbleutt[-1], file=f_txt)
    
    assert len(bubbleutt) == len(oriutt_list)
    f_txt.close()
    with open(os.path.join(out_path, 'bubbleutt.' + suffix + '.pkl'), 'wb') as f:
        pickle.dump(bubbleutt, f)
        

def generate_bubbled_data(bubbleutt_path, out_path, path_prefix_ssd1='', path_prefix_hdd0='', sr=16000):
    with open(os.path.join(path_prefix_hdd0, bubbleutt_path), 'rb') as f:
        bubbleutt_list = pickle.load(f)
    suffix = bubbleutt_path.split('.')[1]
    spk2utt = defaultdict(list)
    spk2uttidx = defaultdict(int)

    for idx, (src_info, noises_info, noises_snrs) in enumerate(bubbleutt_list):
        src_id, src_path, src_start, src_dur = src_info
        speaker_name = src_path.rsplit('/', 1)[1].split('.')[0]
        out_dir = os.path.join(path_prefix_ssd1, out_path, speaker_name)
        os.makedirs(out_dir, exist_ok=True)
        out_name = 'bubble_{}-{}.'.format(speaker_name, spk2uttidx[src_id]) + suffix + '.wav'
        spk2uttidx[src_id] += 1

        abs_src_path = os.path.join(path_prefix_hdd0, src_path)
        src_sig, fs = sf.read(abs_src_path, start=src_start, stop=src_start + src_dur)
        if fs != sr:
            src_sig = librosa.resample(src_sig, fs, sr)
        src_sig = src_sig - np.mean(src_sig)
        src_std = np.std(src_sig)
        src_len = src_sig.shape[-1]
        
        assert len(noises_info) == len(noises_snrs)
        for j, (noise_path, noise_start, noise_dur) in enumerate(noises_info):
            abs_noise_path = os.path.join(path_prefix_hdd0, noise_path)
            noise, fs = sf.read(abs_noise_path, start=noise_start, stop=noise_start + noise_dur)
            if fs != sr:
                noise = librosa.resample(noise, fs, sr)           
            if noise.shape[-1] >= src_len:
                noise = noise[: src_len]
            else:
                noise = np.pad(noise, (0, src_len - noise.shape[-1]), mode='wrap')
            noise = noise - np.mean(noise)
            snr = noises_snrs[j]
            scale_fac = 10**(-snr / 20) * src_std / np.std(noise)
            noise = noise * scale_fac
            src_sig += noise
        
        src_sig = src_sig * src_std / np.std(src_sig)
        sf.write(os.path.join(out_dir, out_name), src_sig, sr)
        spk2utt[src_id].append((os.path.join(out_path, speaker_name, out_name), 0, src_sig.shape[-1]))
        print(idx)

    cnt = 0
    for utt in spk2utt.values():
        cnt += len(utt)
    assert cnt == len(bubbleutt_list)
    assert len(spk2utt.keys()) == 1172
    
    with open(os.path.join(path_prefix_ssd1, out_path, 'spk2utt_bubble.' + suffix + '.pkl'), 'wb') as f:
        pickle.dump(spk2utt, f)


def main(state=0, sub_state=0, path_prefix_hdd0='', path_prefix_ssd1=''):
    if state == 0:
        """ Create spk2utt and oriutt """
        data_path = os.path.join(path_prefix_hdd0, 'DATASET/Librispeech/concatenate_MN/train_clean')
        out_path = os.path.join(path_prefix_hdd0, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA')
        uttlen_limit = [5, 30]
        if sub_state == 0:
            split_concatenate_data(data_path, out_path, uttlen_limit, fs=16000, seed=datetime.datetime.now(), suffix='0')

    if state == 1:
        """ Create revutt """
        oriutt_path = os.path.join(path_prefix_hdd0, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/oriutt.0.pkl')
        rir_set_list_path = [
            os.path.join(path_prefix_hdd0, 'DATASET/RIR_and_noise/simulated_rirs_16k/smallroom/rir_list'),
            os.path.join(path_prefix_hdd0, 'DATASET/RIR_and_noise/simulated_rirs_16k/mediumroom/rir_list'),
        ]
        out_path = os.path.join(path_prefix_hdd0, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA')
        if sub_state == 0:
            assign_rir(oriutt_path, rir_set_list_path, out_path, suffix='0', seed=datetime.datetime.now())

    if state == 2:
        """ Generate reverberated signals and its corresponding spk2utt_rev.pkl """
        revutt_path = 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/revutt.0-0.pkl' 
        rir_path = 'DATASET/RIR_and_noise'
        out_path = 'Librispeech_for_proj/aug_data/dir_aug/reverb'
        if sub_state == 0:
            generate_rev_data(revutt_path, rir_path, out_path, path_prefix_hdd0=path_prefix_hdd0, path_prefix_ssd1=path_prefix_ssd1, sr=16000)

    if state == 3:
        """ Create bubbleutt """
        oriutt_path = os.path.join(path_prefix_hdd0, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/oriutt.0.pkl')
        bubble_path = os.path.join(path_prefix_hdd0, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/spk2utt.0.pkl')
        out_path = os.path.join(path_prefix_hdd0, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA')
        snr_list = [20, 17, 15, 13]
        bubble_num_list = [3, 4, 5, 6, 7]
        if sub_state == 0:
            assign_bubble_noise(oriutt_path, bubble_path, out_path, snr_list=snr_list, bubble_num_list=bubble_num_list,
                                suffix='0', seed=datetime.datetime.now())
        
    if state == 4:
        """ Generate bubbled signals and its corresponding spk2utt_bubble.pkl """
        bubbleutt_path = 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/bubbleutt.0-0.pkl'
        out_path = 'Librispeech_for_proj/aug_data/dir_aug/bubble'
        if sub_state == 0:
            generate_bubbled_data(bubbleutt_path, out_path, path_prefix_hdd0=path_prefix_hdd0, path_prefix_ssd1=path_prefix_ssd1, sr=16000)
            
    if state == 5:
        """ Change audio path in spk2utt """
        file_path = os.path.join(path_prefix_hdd0, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/SEP_AUG_DATA/combine/egs/egs.diagnosis.pkl')
        target_path = 'Librispeech_for_proj/aug_data/dir_aug/bubble'
        level = 0
        ori_preserve = 2
        del_ori = False
        if sub_state == 0:
            change_path(file_path, target_path, level=level, ori_preserve=ori_preserve, del_ori=del_ori)


if __name__ == "__main__":
    path_prefix_hdd0 = ['/home/user/zhaoyi.gu/mnt/g2', '/data/hdd0/zhaoyigu'][1]
    path_prefix_ssd1 = ['/home/user/zhaoyi.gu/mnt/g2/ssd1', '/data/ssd1/zhaoyi.gu'][1]
    main(state=5, sub_state=0, path_prefix_hdd0=path_prefix_hdd0, path_prefix_ssd1=path_prefix_ssd1)
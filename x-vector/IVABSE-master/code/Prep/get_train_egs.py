import os
import pickle
import random
import copy
import datetime
import numpy as np
import soundfile as sf
from collections import defaultdict
from utils import get_uttnum, get_egsnum, change_path


def combine_spk2utt(out_filepath, *args):
    """ Combine multiple spk2utt.*.pkl file into one """
    """ 
    Args:
        `out_filename`: [str] filepath where output will be saved
        `args`: [tuple] tuple consists of all the absolute filename to spk2utt.pkl that needs to be combined

    Out:
        `spk2utt`: [Not returned but saved][dict]
                       dict object that contains spk2utt information for combined dataset
    """
    os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

    suffix = os.path.basename(out_filepath).split('.')[1]
    f_log = open(os.path.join(os.path.dirname(out_filepath), 'combine.' + suffix + '.log'), 'at')
    spk2utt_all = []
    for in_path in args:
        f_in = open(in_path, 'rb')
        spk2utt_part = pickle.load(f_in)
        f_in.close()
        spk2utt_all.append(spk2utt_part)
    
    spk2utt = {}
    for i, spk2utt_part in enumerate(spk2utt_all):
        spk2utt_part_uttnum = 0
        for spkr_id in spk2utt_part:
            if spkr_id not in spk2utt:
                spk2utt[spkr_id] = []
            spk2utt[spkr_id].extend(spk2utt_part[spkr_id])
            spk2utt_part_uttnum += len(spk2utt_part[spkr_id])
        print('Total utterance number for {} is {}'.format(args[i], spk2utt_part_uttnum), file=f_log)
        print('Finish processing file {}'.format(args[i]))

    spk2utt_uttnum = 0
    for spkr_id in spk2utt: 
        spk2utt_uttnum += len(spk2utt[spkr_id])
    print('Total utterance number for final spk2utt.pkl is {}'.format(spk2utt_uttnum), file=f_log)
    print('\n', file=f_log)
    f_log.close()

    with open(out_filepath, 'wb') as f_spk2utt: 
        pickle.dump(spk2utt, f_spk2utt)


def split_dataset(datainfo_path, out_path, uttnum_tosplit, namestr_for_remain, namestr_for_split, exclude=True, min_utt=8, prefix_level=1):
    """ Split dataset into two parts """
    """ 
    Args:
        `datainfo_path`: path where spk2utt.pkl is saved
        `out_path`: path where spk2utt_rem.pkl and spk2utt_seg.pkl will be saved
        `uttnum_tosplit`: number of utterances selected for segmented spk2utt_seg
        `namestr_for_remain`: filename for spk2utt_rem i.e. namestr_for_remain.pkl
        `namestr_for_split`: filename for spk2utt_seg i.e. namestr_for_split.pkl
        `exclude`: [bool] if True, spk2utt_rem and spk2utt_seg have no common utterances;
                           if False, spk2utt_rem contains all contents of the original spk2utt
        `min_utt`: when number of utterances for some speaker is smaller than this value, then the speaker won't be splited 
    
    Out:
        `spk2utt_rem`: [Not returned but saved][dict]
                       dict object containing spk2utt information for remaining dataset
        `spk2utt_seg`: [Not returned but saved][dict]
                       dict object containing spk2utt information for segmented dataset
        `namestr_for_remain.pkl`: pickle file that saves spk2utt_rem
        `namestr_for_split.pkl`: pickle file that saves spk2utt_seg
    """
    os.makedirs(out_path, exist_ok=True)
    random.seed(datetime.datetime.now())

    f_spk2utt = open(os.path.join(datainfo_path), 'rb')
    spk2utt = pickle.load(f_spk2utt)
    f_spk2utt.close()
    f_txt = open(os.path.join(out_path, 'dataset_split-' + os.path.basename(datainfo_path).rsplit('.', 1)[0] + '.txt'), 'wt')
    print("datainfo_path: {}".format(datainfo_path), file=f_txt)
    print("out_path: {}".format(out_path), file=f_txt)
    print("uttnum_tosplit: {}".format(uttnum_tosplit), file=f_txt)
    print("namestr_for_remain: {}".format(namestr_for_remain), file=f_txt)
    print("namestr_for_split: {}".format(namestr_for_split), file=f_txt)
    print("exclude: {}".format(exclude), file=f_txt)
    print("min_utt: {}".format(min_utt), file=f_txt)
    f_txt.close()

    spkr_list = list(spk2utt.keys())
    uttnum = get_uttnum(spk2utt)
    if uttnum_tosplit > sum(uttnum):
        raise ValueError('Number of utterances to split exceeds number of utterances in spk2utt.pkl')

    f_spk2utt_rem = open(os.path.join(out_path, os.path.basename(datainfo_path).rsplit('.', prefix_level)[0] + '.' + namestr_for_remain + '.pkl'), 'wb')
    f_spk2utt_seg = open(os.path.join(out_path, os.path.basename(datainfo_path).rsplit('.', prefix_level)[0] + '.' + namestr_for_split + '.pkl'), 'wb')
    spk2utt_rem = copy.deepcopy(spk2utt) 
    spk2utt_seg = defaultdict(list) 
    splituttcnt = 0
    
    while True:
        if splituttcnt == uttnum_tosplit:
            break
        random.shuffle(spkr_list)
        for i, spkr in enumerate(spkr_list):
            if (len(spk2utt[spkr]) < min_utt and exclude is True) or len(spk2utt_rem[spkr]) == 0:
                continue
            selected = random.choice(spk2utt_rem[spkr])
            spk2utt_seg[spkr].append(selected)
            spk2utt_rem[spkr].remove(selected)
            splituttcnt += 1
            if splituttcnt == uttnum_tosplit:
                break
    
    pickle.dump(spk2utt_seg, f_spk2utt_seg)
    if exclude:
        pickle.dump(spk2utt_rem, f_spk2utt_rem)
    else:
        pickle.dump(spk2utt, f_spk2utt_rem)
    f_spk2utt_rem.close()
    f_spk2utt_seg.close()


def _check_spkr(spk2utt, min_nutt_per_spkr):
    """ check """

    deprecated_spkrs = []
    max_key = -1
    for key, value in spk2utt.items():
        if int(key) > max_key:
            max_key = key
        if len(value) < min_nutt_per_spkr:
            deprecated_spkrs.append(int(key))
    assert len(deprecated_spkrs) == 0, 'Bad speaker still exists!'
    assert len(spk2utt) == max_key + 1, 'Speaker number and speaker id does not match!'


def _deprecate_spkr(deprecated_spkrs, key_map, spk2utt):
    """ delete deprecated_spkr from spk2utt, adjust spkr_id afterwards """
    """ 
    Args:
        deprecated_spkrs: [list] bad speaker to be deleted
        key_map: [dict] a map that maps original key to current key
        spk2utt: [dict] dict whose key in bad_spkr needs to be deleted and other keys adjusted
    """
    key_list = list(spk2utt.keys())
    key_list = sorted(key_list)
    for spkr_id in key_list:
        if spkr_id in deprecated_spkrs:
            del spk2utt[spkr_id]
        else:
            assert key_map[spkr_id] <= spkr_id
            spk2utt[key_map[spkr_id]] = spk2utt.pop(spkr_id)


def check_num_utt(datainfo_filepath, min_nutt_per_spkr):
    """ Remove spkr whose utterances are less than min_nutt_per_spkr, adjust spkr_id afterwards """
    with open(datainfo_filepath, 'rb') as f_spk2utt:
        spk2utt = pickle.load(f_spk2utt)
    suffix = os.path.basename(datainfo_filepath).split('.')[1]
    f_txt = open(os.path.join(os.path.dirname(datainfo_filepath), 'deprecate.' + suffix + '.log'), 'at')
    print("datainfo_filepath: {}".format(datainfo_filepath), file=f_txt)
    print("min_nutt_per_spkr: {}".format(min_nutt_per_spkr), file=f_txt)

    deprecated_spkrs = []
    for key, value in spk2utt.items():
        if len(value) < min_nutt_per_spkr:
            deprecated_spkrs.append(int(key))
    deprecated_spkrs = sorted(deprecated_spkrs, reverse=True)
    deprecated_spkrs_dur = [sum([x[-1] for x in spk2utt[spkr]]) / 16000 / 60 for spkr in deprecated_spkrs]
    print('These speakers need to be deleted in {}: {}'.format(datainfo_filepath, deprecated_spkrs))
    print('These speakers need to be deleted in {}: {}'.format(datainfo_filepath, deprecated_spkrs), file=f_txt)
    print('Their total duration is {} minutes'.format(deprecated_spkrs_dur), file=f_txt)

    key_map = {}
    for key in spk2utt.keys():
        downsize = sum([int(int(key) > bad_spkr) for bad_spkr in deprecated_spkrs])
        key_map[key] = key - downsize
    _deprecate_spkr(deprecated_spkrs, key_map, spk2utt) 

    _check_spkr(spk2utt, min_nutt_per_spkr)
    print('Successfully deleted these speaker in {}: {}'.format(datainfo_filepath, deprecated_spkrs))
    print('Successfully deleted these speaker in {}: {}'.format(datainfo_filepath, deprecated_spkrs), file=f_txt)
    print('\n', file=f_txt)

    with open(datainfo_filepath.rsplit('.', 1)[0] + '.chosen.pkl', 'wb') as f_spk2utt:
        pickle.dump(spk2utt, f_spk2utt)
    with open(os.path.join(os.path.dirname(datainfo_filepath), 'deprecated_key_map.' + suffix + '.pkl'), 'wb') as f:
        pickle.dump((deprecated_spkrs, key_map), f)
    f_txt.close()
    
    return deprecated_spkrs, key_map


def check_extra_spkr(datainfo_filepath, deprecated_spkrs=None, key_map=None, key_map_filepath=None):
    """ Remove extra spkrs from diagnosis/vad set """
    with open(datainfo_filepath, 'rb') as f:
        spk2utt = pickle.load(f)

    if key_map_filepath is not None:
        with open(key_map_filepath, 'rb') as f:
            deprecated_spkrs, key_map = pickle.load(f)
    
    _deprecate_spkr(deprecated_spkrs, key_map, spk2utt)
    with open(datainfo_filepath.rsplit('.', 1)[0] + '.chosen.pkl', 'wb') as f:
        pickle.dump(spk2utt, f) 


def allocate_egs(datainfo_filepath, out_filepath, nframe_per_ark, n_ark, negs_per_ark, random_chunklen=True, seed=None, **kwargs):
    """ Pick specific n_ark * negs_per_ark examples from spk2utt.pkl """
    """ 
    Args:
        `datainfo_filepath`: [str] filepath to spk2utt.pkl
        `out_filepath`: [str] filepath where out_filename will be saved
        `nframe_per_ark`: [2-element list] 2-element list: contains minimum and maximum number of frames per archive in ascending order
        `n_ark`: [int] number of archives
        `negs_per_ark`: [int] number of examples in each archive
        `kwargs`: [dict] containing 'fs', 'nfft', 'hop'

    Out:
        `all_egs`: [Not returned but saved][list]
                   [{spkr-id-0: [(uttpath, offset, chunklen), (), ...]}, {archive-1}, {archive-2}, ...]
        `out_filepath`: pickle file that saves all_egs
    """

    os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
    fs = kwargs['fs']
    hop = int(kwargs['stft_hop'] * fs)
    random.seed(seed)

    def get_chunk_len(nframe_per_ark, n_ark, random_chunklen):
        if random_chunklen:
            return [random.randint(nframe_per_ark[0], nframe_per_ark[1]) for _ in range(n_ark)]
        else:
            return [int(round(n)) for n in np.linspace(nframe_per_ark[0], nframe_per_ark[1], num=n_ark)]

    f_spk2utt = open(datainfo_filepath, 'rb')
    spk2utt = pickle.load(f_spk2utt)
    f_spk2utt.close()
    spkr_list = list(spk2utt.keys())
    chunklen_list = get_chunk_len(nframe_per_ark, n_ark, random_chunklen)
    random.shuffle(chunklen_list)

    all_egs = []
    for ark_id in range(n_ark):
        random.shuffle(spkr_list)
        chunk_len = int((chunklen_list[ark_id] - 1) * hop)
        processed_egs = 0
        these_egs = {}
        while True:
            if processed_egs == negs_per_ark:
                break
            for spkr in spkr_list:
                if spkr not in these_egs:
                    these_egs[spkr] = []
                uttpath, startpoint, dur = random.choice(spk2utt[spkr])  # randomly pick a utterance
                assert dur > chunk_len
                # TODO: when duration < chunk_len
                offset = startpoint + random.randint(0, dur - chunk_len)  # randomly pick an offset
                these_egs[spkr].append((uttpath, offset, chunk_len))
                processed_egs += 1
                if processed_egs == negs_per_ark:
                    break
        print('Finish generating examples for archive {}'.format(ark_id))
        all_egs.append(these_egs)
    
    with open(out_filepath, 'wb') as f_all_egs:
        pickle.dump(all_egs, f_all_egs)
        

def concate_egs(datapath, out_filepath):
    """ Called when need to concatenate examples into a long wav file """
    os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

    map_dict = {}
    spkr_list = list(filter(lambda x: x.startswith('speaker'), os.listdir(datapath)))
    for spkr_file in spkr_list:
        if spkr_file not in map_dict:
            map_dict[spkr_file] = {}
        wav_list = list(filter(lambda x: x.endswith('wav'), os.listdir(os.path.join(datapath, spkr_file))))
        src = []
        offset = 0
        for wav_file in wav_list:
            x, sr = sf.read(os.path.join(datapath, spkr_file, wav_file))
            src.append(x)
            map_dict[spkr_file][wav_file] = (offset, x.shape[0])
            offset += x.shape[0]
        src = np.concatenate(src, axis=0)
        sf.write(os.path.join(os.path.dirname(out_filepath), spkr_file + '.wav'), src, sr)
        print('Finish processing {}'.format(spkr_file))
    with open(out_filepath, 'wb') as f:
        pickle.dump(map_dict, f)


def main(state=0, sub_state=0, *, path_prefix='/data/hdd0/zhaoyigu'):

    if state == 0:
        """ Combine spk2utt.{}.pkl into spk2utt_sep.pkl """
        out_filepath = path_prefix + '/DATASET/SEPARATED_LIBRISPEECH/DATA_MN/train_clean/spk2utt_sep.pkl'
        njobs = 12
        args = [path_prefix + '/DATASET/SEPARATED_LIBRISPEECH/DATA_MN/train_clean/spk2utt.{}.pkl'.format(n) for n in range(njobs)]
        args = tuple(args)
        if sub_state == 0:
            combine_spk2utt(out_filepath, *args)
        else:
            f = open(out_filepath, 'rb')
            spk2utt = pickle.load(f)
            uttnum = get_uttnum(spk2utt)
            max_utt, min_utt = 0, float("inf")
            for spkr in spk2utt.keys():
                if len(spk2utt[spkr]) > max_utt:
                    max_utt = len(spk2utt[spkr])
                if len(spk2utt[spkr]) < min_utt:
                    min_utt = len(spk2utt[spkr])
            f.close()

    elif state == 1:
        """ Split spk2utt.pkl into train, vad, diagnosis """
        if sub_state == 0:
            """ Split dataset from spk2utt.pkl and spk2utt_*.pkl to train and vad """
            datainfo_path = [
                os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/spk2utt_clean.0.pkl'),
                os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/spk2utt_rev.0-0.pkl'),
                os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/spk2utt_bubble.0-0.pkl'),
            ]
            namestr_for_remain = ['train', 'train', 'train']
            namestr_for_split = ['vad', 'vad', 'vad']
            uttnum_tosplit = [int(1172 * 4), int(1172 * 4), int(1172 * 4)]
            exclude = [True, True, True]
            min_utt = [8, 8, 8]
            prefix_level = [1, 1, 1]
            out_path = [os.path.dirname(path) for path in datainfo_path]
            for i in range(len(datainfo_path)):
                split_dataset(datainfo_path[i], out_path[i], uttnum_tosplit[i], namestr_for_remain[i], namestr_for_split[i],
                              exclude=exclude[i], min_utt=min_utt[i], prefix_level=prefix_level[i])

        elif sub_state == 1:
            """ Split dataset from spk2utt_train.pkl and spk2utt_*_train.pkl to train and diagnosis  """ 
            datainfo_path = [
                os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/spk2utt_clean.0.train.pkl'),
                os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/spk2utt_rev.0-0.train.pkl'),
                os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/spk2utt_bubble.0-0.train.pkl'),
            ]
            namestr_for_remain = ['train', 'train', 'train']
            namestr_for_split = ['diagnosis', 'diagnosis', 'diagnosis']
            uttnum_tosplit = [int(1172 * 4), int(1172 * 4), int(1172 * 4)]
            exclude = [False, False, False]
            min_utt = [8, 8, 8]
            prefix_level = [2, 2, 2]
            out_path = [os.path.dirname(path) for path in datainfo_path]
            for i in range(len(datainfo_path)):
                split_dataset(datainfo_path[i], out_path[i], uttnum_tosplit[i], namestr_for_remain[i], namestr_for_split[i],
                              exclude=exclude[i], min_utt=min_utt[i], prefix_level=prefix_level[i])
        
        elif sub_state == 2:
            """ Combine spk2utt.train/vad/diagnosis.pkl with spk2utt_*.train/vad/diagnosis.pkl """
            suffix_list = ['train', 'vad', 'diagnosis']
            for suffix in suffix_list:
                out_filepath = os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/combine/spk2utt.0-0.' + suffix + '.pkl')
                args = [
                    os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/spk2utt_clean.0.' + suffix + '.pkl'),
                    os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/spk2utt_rev.0-0.' + suffix + '.pkl'),
                    os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/spk2utt_bubble.0-0.' + suffix + '.pkl'),
                ]
                combine_spk2utt(out_filepath, *args)

    elif state == 2:
        """ Deprecate speakers which do not meet requirements """
        if sub_state == 0:
            """ Deprecate speaker whose utterance number is not adequate """
            datainfo_filepath = path_prefix + '/DATASET/Librispeech/concatenate/train_clean/spk2utt_train.pkl'
            deprecated_spkrs, map_key = check_num_utt(datainfo_filepath, 12)

        elif sub_state == 1:
            """ Deprecate and adjust spkr_id in test, vad according to deprecated_key_map.pkl information """
            key_map_filepath = os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/combine/deprecated_key_map.pkl')
            datainfo_filepath = os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/combine/spk2utt.0-0.diagnosis.pkl')
            check_extra_spkr(datainfo_filepath, key_map_filepath=key_map_filepath)
        
        elif sub_state == 2:
            datainfo_filepath_main_aug = os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/combine/spk2utt.0-0.train.chosen.pkl')
            datainfo_filepath_patient1_aug = os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/combine/spk2utt.0-0.vad.chosen.pkl')
            datainfo_filepath_patient2_aug = os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/combine/spk2utt.0-0.diagnosis.chosen.pkl')
            datainfo_filepath_main = os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/combine/spk2utt.0-0.train.pkl')
            datainfo_filepath_patient1 = os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/combine/spk2utt.0-0.vad.pkl')
            datainfo_filepath_patient2 = os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/combine/spk2utt.0-0.diagnosis.pkl')
            with open(datainfo_filepath_main_aug, 'rb') as f:
                spk2utt_main_aug = pickle.load(f)
            with open(datainfo_filepath_patient1_aug, 'rb') as f:
                spk2utt_pat1_aug = pickle.load(f)
            with open(datainfo_filepath_patient2_aug, 'rb') as f:
                spk2utt_pat2_aug = pickle.load(f)
            with open(datainfo_filepath_main, 'rb') as f:
                spk2utt_main = pickle.load(f)
            with open(datainfo_filepath_patient1, 'rb') as f:
                spk2utt_pat1 = pickle.load(f)
            with open(datainfo_filepath_patient2, 'rb') as f:
                spk2utt_pat2 = pickle.load(f)
            dummy = 1

    elif state == 3:
        kwargs = {}
        kwargs['fs'] = 16000
        kwargs['stft_len'] = 0.064
        kwargs['stft_hop'] = 0.016

        if sub_state == 0:
            """ Generate egs for spk2utt_train.pkl """
            datainfo_filepath = os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/combine/spk2utt.0-0.train.chosen.pkl')
            out_filepath = os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/combine/egs/egs.train.pkl')
            nframe_per_ark = [160, 200]
            n_ark = 9
            negs_per_ark = round(3000000 / 9)
            allocate_egs(datainfo_filepath, out_filepath, nframe_per_ark, n_ark, negs_per_ark,
                         random_chunklen=False, seed=datetime.datetime.now(), **kwargs)
        
        elif sub_state == 1:
            """ Generate egs for spk2utt_vad.pkl """
            datainfo_filepath = os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/combine/spk2utt.0-0.vad.chosen.pkl')
            out_filepath = os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/combine/egs/egs.vad.pkl')
            nframe_per_ark = [160, 200]
            n_ark = 9
            negs_per_ark = round(330000 / 9)
            allocate_egs(datainfo_filepath, out_filepath, nframe_per_ark, n_ark, negs_per_ark, random_chunklen=False, **kwargs)
    
        elif sub_state == 2:
            """ Generate egs for spk2utt_diagnosis.pkl """
            datainfo_filepath = os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/combine/spk2utt.0-0.diagnosis.chosen.pkl')
            out_filepath = os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/combine/egs/egs.diagnosis.pkl')
            nframe_per_ark = [160, 200]
            n_ark = 9
            negs_per_ark = round(330000 / 9)
            allocate_egs(datainfo_filepath, out_filepath, nframe_per_ark, n_ark, negs_per_ark, random_chunklen=False, **kwargs)
    
    elif state == 4:
        """ Help determine negs_per_ark """
        # info_path = path_prefix + '/DATASET/Librispeech/concatenate/train_clean/info.pkl'
        # info_path_2 = path_prefix + '/DATASET/Librispeech/concatenate_MN/train_clean/info.pkl'
        # with open(info_path, 'rb') as f:
        #     info = pickle.load(f)
        # with open(info_path_2, 'rb') as f:
        #     info_2 = pickle.load(f)
        # for i in range(len(info)):
        #     assert info[i][0].rsplit('/', maxsplit=1)[1] == info_2[i][0].rsplit('/', maxsplit=1)[1] 
        #     assert info[i][1] == info_2[i][1]
        #     assert info[i][2] == info_2[i][2]

        # all_nsample = sum([x[1] for x in info])
        # nsample = (160 - 1) * 256
        # hop = int(nsample * 1)
        # negs = 0
        # for spkr in info:
        #     negs += int((spkr[1] - nsample) // hop + 1)
        
        spk2utt_path = os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/combine/spk2utt.0-0.vad.chosen.pkl')
        with open(spk2utt_path, 'rb') as f:
            spk2utt = pickle.load(f)
        nsample = (180 - 1) * 256
        hop = int(nsample * 0.25)
        negs = 0
        for spkr in spk2utt.keys():
            nsamp = sum([x[2] for x in spk2utt[spkr]])
            negs += int((nsamp - nsample) // hop + 1)
    
        # egs_path = path_prefix + '/DATASET/Librispeech/concatenate/train_clean/egs/egs_train.pkl' 
        # with open(egs_path, 'rb') as f:
        #     egs = pickle.load(f)
        # negs = get_egsnum(egs)

        print(negs)
    
    elif state == 5:
        datapath = os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/DATA/train_clean/')
        out_filepath = os.path.join(path_prefix, 'DATASET/SEPARATED_LIBRISPEECH/DATA/train_clean/concatenate/map_dict.pkl')
        concate_egs(datapath, out_filepath)


if __name__ == "__main__":
    path_prefix = ['/home/user/zhaoyi.gu/mnt/g2', '/data/hdd0/zhaoyigu'][1]
    main(state=3, sub_state=2, path_prefix=path_prefix)


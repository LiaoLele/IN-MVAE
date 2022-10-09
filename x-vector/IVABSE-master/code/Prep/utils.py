import re
import os
import pickle
import copy
import numpy as np
import random


def get_uttnum(spk2utt):
    """ calculate number of utterances in spk2utt """
    """ 
    Args:
        spk2utt: [dict]
    Out:
        num: [returned][int]
             number of utterances in spk2utt
    """
    num = []
    for value in spk2utt.values():
        num.append(len(value))
    return num


def get_egsnum(egs):
    num = 0
    for ark in egs:
        num += get_uttnum(ark)
    return num


def assign_rir(rirfile_path, targetfile_path, outfile_path, assign_key=None, mode=0):
    """ assign one rir to each utterance pair for mixture simulation """
    """ 
    Args:
        `rirfile_path`: [str] path where xxx2gidx.pkl is saved
        `targetfile_path`: [str] path where mix2pair.pkl is saved
        `outfile_path`: [str] path where makemix.pkl will be saved
        `assign_key`: [str/float/int] one specific key in xxx2gidx.pkl 
    Out:
        `makemix`: [Not returned but saved][list]
                    [[(spkr-id-0-a, datapath, start-0-a, dur-0-a), (spkr-id-0-b, datapath, start-0-b, dur-0-b), global-idx-for-rir], [(), (), int], ...]
        `makemix.pkl`: pickle file that saves makemix
    """
    os.makedirs(os.path.dirname(outfile_path), exist_ok=True)

    f_key2gidx = open(rirfile_path, 'rb')
    f_mix2pair = open(targetfile_path, 'rb')
    key2gidx = pickle.load(f_key2gidx)
    mix2pair = pickle.load(f_mix2pair)
    f_key2gidx.close()
    f_mix2pair.close()

    f_makemix = open(outfile_path, 'wb')

    pair_num = len(mix2pair)
    print('In total, there are {} pairs to be mixed.'.format(pair_num))
    if assign_key is None:
        pair_idx = 0
        while True:
            for key in key2gidx.keys():
                if pair_idx == pair_num:
                    break
                rir_idx = random.choice(key2gidx[key])
                mix2pair[pair_idx].append(rir_idx)
                pair_idx += 1
            if pair_idx == pair_num:
                break
    else:
        if mode == 0:
            """ key2gidx: {"0.15-20": [idx-0, idx-1,...], ...} """
            """ for test_simu """
            for pair_idx in range(pair_num):
                rir_idx = random.choice(key2gidx[assign_key])
                mix2pair[pair_idx].append(rir_idx)
        elif mode == 1:
            """ key2gidx: {"0.160": [[angle-interval=20], [angle-interval=30], ...]} """
            """ for test_real_2_src """
            pair_idx = 0
            while True:
                for rir_idx_list in key2gidx[assign_key]:
                    if pair_idx == pair_num:
                        break
                    rir_idx = random.choice(rir_idx_list)
                    mix2pair[pair_idx].append(rir_idx)
                    pair_idx += 1
                if pair_idx == pair_num:
                    break
        elif mode == 2:
            """ key2gidx: {"0.160": [idx-0, idx-1, ...], ...} """
            """ for test_real_4_src """
            pair_idx = 0
            while True:
                for rir_idx in key2gidx[assign_key]:
                    if pair_idx == pair_num:
                        break
                    mix2pair[pair_idx].append(rir_idx)
                    pair_idx += 1
                if pair_idx == pair_num:
                    break
                    
    pickle.dump(mix2pair, f_makemix)
    f_makemix.close()


def assign_sir(targetfile_path, outfile_path, sir_range=None, use_random=False, make_order=False):
    os.makedirs(os.path.dirname(outfile_path), exist_ok=True)
    with open(targetfile_path, 'rb') as f:
        makemix = pickle.load(f)
    pair_num = len(makemix)
    if use_random:
        for idx in range(pair_num):
            makemix[idx].append(random.choice(sir_range))
            if make_order:
                src_num = len(makemix[idx]) - 2
                sir_order = list(range(src_num))
                random.shuffle(sir_order)
                makemix[idx].append(sir_order)
    else:
        pair_idx = 0
        while True:
            for sir in sir_range:
                if pair_idx == pair_num:
                    break
                makemix[pair_idx].append(sir)
                if make_order:
                    src_num = len(makemix[pair_idx]) - 2
                    sir_order = list(range(src_num))
                    random.shuffle(sir_order)
                    makemix[pair_idx].append(sir_order)
                pair_idx += 1
            if pair_idx == pair_num:
                break
    with open(outfile_path, 'wb') as f:
        pickle.dump(makemix, f)


def view_info_file(file_path):
    with open(file_path, 'rb') as f:
        info = pickle.load(f)
    info = sorted(info, key=lambda x: x[1])
    dummy = 1
    

def mysort(x):
    x = sorted(list(enumerate(x)), key=lambda x: x[1])
    index, x = list(zip(*x))
    return index, x
    

def zero_pad(x, num_pad, hop_length=256):
    """ x: [n_channel, nsample] """
    if (x.shape[1] / hop_length + 1) % num_pad == 0:    # common equation is (x.shape[1]/frame_move + 1) % num_pad == 0, where num_pad is the required to conduct cnn
        return x
    rest = (num_pad * hop_length) - (x.shape[1] + hop_length) % (num_pad * hop_length)
    left = rest // 2
    right = rest - left
    return np.pad(x, ((0, 0), (left, right)), mode='constant')


def change_path(file_path, target_path, ori_preserve=1, level=0, del_ori=False):
    """ For changing path info in spk2utt.pkl or *utt.pkl """
    """ 
    Args:
        `file_path`: path where *.pkl file is saved
        `target_path`: new common path prefix 
        `ori_preserve`: number of '/' to rsplit(), so that except the first split part, the rest are concatenated and joined with target_pagh 
        `level`: the idx of path parameter in a list
        `del_ori`: whether to delete the old *.pkl
    """
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    
    ptn = re.compile(r'^.*DATASET/(?P<rel_path>.*)$')
    
    if isinstance(obj, dict):
        for key in obj.keys():
            for idx, item in enumerate(obj[key]):
                item = list(item)
                preserved_path = "/".join(item[level].rsplit('/', ori_preserve)[1:])
                item[level] = os.path.join(target_path, preserved_path)
                obj[key][idx] = tuple(item)
    
    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            if not isinstance(item, dict):
                item = list(item)
                item[level] = os.path.join(target_path, os.path.basename(item[level]))
                obj[idx] = tuple(item)
            else:
                for key in item.keys():
                    for inneridx, inneritem in enumerate(item[key]):
                        inneritem = list(inneritem)
                        path = inneritem[0]
                        if ptn.match(path)['rel_path'].startswith('Librispeech'):
                            inneritem[0] = os.path.join('Librispeech_for_proj/ori_data', ptn.match(path)['rel_path'].rsplit('/', maxsplit=1)[1])
                        else:
                            inneritem[0] = os.path.join('Librispeech_for_proj/aug_data/sep_aug/sep',
                                                ptn.match(path)['rel_path'].rsplit('/', maxsplit=2)[1],
                                                ptn.match(path)['rel_path'].rsplit('/', maxsplit=2)[2])
                        # preserved_path = "/".join(inneritem[level].rsplit('/', ori_preserve)[1:])
                        # inneritem[level] = os.path.join(target_path, preserved_path)
                        item[key][inneridx] = tuple(inneritem)

    file_path_split = file_path.rsplit('.', 1)
    os.system("mv {} {}".format(file_path, file_path_split[0] + '.bak.' + file_path_split[1]))
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    if del_ori:
        os.system("rm {}".format(file_path_split[0] + '.bak.' + file_path_split[1]))
    

if __name__ == "__main__":
    librispeech_path = '/home/user/zhaoyi.gu/mnt/g4/LibriSpeech/test-clean'
    test_dev_path = '/data/hdd0/zhaoyigu/DATASET/Librispeech/concatenate/dev_clean'
    out_path = '/data/hdd0/zhaoyigu/DATASET/Librispeech/concatenate/test_clean'
    # sep(librispeech_path, test_dev_path, out_path)
    info_path = '/data/hdd0/zhaoyigu/DATASET/Librispeech/concatenate/test_clean/info.pkl' 
    view_info_file(info_path)

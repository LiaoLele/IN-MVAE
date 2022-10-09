import os
import pickle
import random
import copy
import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
from utils import get_key2utts, get_key2lens


def MakeSubset(datainfo_path, out_dir, out_namestr,
               max_utt_per_key=None, utt_keep_mode='max', 
               max_key_num=None, key_keep_mode='max',
               seed=None):
    os.makedirs(out_dir, exist_ok=True)
    if seed is None:
        seed = datetime.datetime.now()
    random.seed(seed)
    # datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/spk2utt.clean.pkl'
    with open(datainfo_path, 'rb') as f:
        spk2utt = pickle.load(f)
    ori_basename = os.path.basename(datainfo_path).rsplit('.', 1)
    ori_basename.insert(1, out_namestr)
    out_basename = '.'.join(ori_basename)

    if max_key_num is not None:
        new_spk2utt = defaultdict(list)
        assert max_key_num <= len(spk2utt.keys())
        key2lens = get_key2lens(spk2utt)
        key2len2list = [(key, val) for key, val in key2lens.items()]
        if key_keep_mode == 'max':
            key2len2list = sorted(key2len2list, key=lambda x: x[1], reverse=True)[: max_key_num]
        elif key_keep_mode == 'min':
            key2len2list = sorted(key2len2list, key=lambda x: x[1], reverse=False)[: max_key_num]
        elif key_keep_mode == 'random':
            key2len2list = random.shuffle(key2len2list)[:max_key_num]
        targetkeys = list(zip(*key2len2list))[0]
        for key in targetkeys:
            new_spk2utt[key] = copy.deepcopy(spk2utt[key])
    else:
        new_spk2utt = copy.deepcopy(spk2utt)
    
    if max_utt_per_key is not None:
        key2utts = get_key2utts(new_spk2utt)
        uttnums = [(key, uttnum) for key, uttnum in key2utts.items()]
        uttnums = list(filter(lambda x: x[1] > max_utt_per_key, uttnums))
        targetkeys = list(zip(*uttnums))[0]
        for key in targetkeys:
            utts = copy.deepcopy(new_spk2utt[key])
            uttlen_list = [(idx, length) for idx, (_, _, length) in enumerate(utts)]
            if utt_keep_mode == 'max':
                uttlen_list = sorted(uttlen_list, key=lambda x: x[1], reverse=True)[: max_utt_per_key]
            elif utt_keep_mode == 'min':
                uttlen_list = sorted(uttlen_list, key=lambda x: x[1], reverse=False)[: max_utt_per_key]
            elif utt_keep_mode == 'random':
                uttlen_list = random.shuffle(uttlen_list)[: max_utt_per_key]
            targetutts = list(zip(*uttlen_list))[0]
            new_spk2utt[key] = [utts[idx] for idx in targetutts]

    with open(os.path.join(out_dir, out_basename), 'wb') as f:
        pickle.dump(new_spk2utt, f)

    
def SplitDataset(datainfo_path, out_path, uttnum_tosplit, namestr_for_remain, namestr_for_split,
                 exclude=True, min_utt=8,
                 seed=None):
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
    if seed is None:
        seed = datetime.datetime.now()
    random.seed(seed)

    namestr_list = os.path.basename(datainfo_path).rsplit('.', 1)
    with open(os.path.join(datainfo_path), 'rb') as f_spk2utt:
        spk2utt = pickle.load(f_spk2utt)
    f_txt = open(os.path.join(out_path, f"Readme-SplitDataset.{os.path.basename(datainfo_path).split('.')[2]}.txt"), 'wt')
    basename_for_remain = namestr_list.copy()
    basename_for_split = namestr_list.copy()
    basename_for_remain.insert(1, namestr_for_remain)
    basename_for_split.insert(1, namestr_for_split)
    basename_for_remain = '.'.join(basename_for_remain)
    basename_for_split = '.'.join(basename_for_split)
    
    print("Dataset path to split: {}".format(datainfo_path), file=f_txt)
    print("Dataset path after split: {}".format(out_path), file=f_txt)
    print("Dataset namestr for remain: {}".format(basename_for_remain), file=f_txt)
    print("Dataset namestr for split: {}".format(basename_for_split), file=f_txt)
    print("Total number of utterance to split: {}".format(uttnum_tosplit), file=f_txt)
    print("The remaining dataset does not include those in the splited dataset: {}".format(exclude), file=f_txt)
    print("Minimum number of utterance in the remained dataset for each key after split: {}".format(min_utt), file=f_txt)

    spkr_list = list(spk2utt.keys())
    key2utts = get_key2utts(spk2utt)
    uttnum = list(key2utts.values())
    if uttnum_tosplit > sum(uttnum):
        raise ValueError('Number of utterances to split exceeds number of utterances in spk2utt.pkl')

    spk2utt_rem = copy.deepcopy(spk2utt) 
    spk2utt_seg = defaultdict(list) 
    splituttcnt = 0
    freeze_spkers = 0
    while True:
        if splituttcnt == uttnum_tosplit:
            break
        if freeze_spkers == len(spk2utt_rem.keys()):
            raise ValueError('Cannot split more utterance!')
        freeze_spkers = 0
        random.shuffle(spkr_list)
        for i, spkr in enumerate(spkr_list):
            if (len(spk2utt_rem[spkr]) < min_utt and exclude is True):
                freeze_spkers += 1
                continue
            selected = random.choice(spk2utt_rem[spkr])
            spk2utt_seg[spkr].append(selected)
            spk2utt_rem[spkr].remove(selected)
            splituttcnt += 1
            if splituttcnt == uttnum_tosplit:
                break
    
    print(f"Number of keys in {basename_for_remain} is {len(spk2utt_rem.keys())}", file=f_txt)
    print(f"Number of keys in {basename_for_split} is {len(spk2utt_seg.keys())}", file=f_txt)
    f_txt.close()
    with open(os.path.join(out_path, basename_for_split), 'wb') as f_spk2utt_seg:
        pickle.dump(spk2utt_seg, f_spk2utt_seg)
    with open(os.path.join(out_path, basename_for_remain), 'wb') as f_spk2utt_rem:
        if exclude:
            pickle.dump(spk2utt_rem, f_spk2utt_rem)
        else:
            pickle.dump(spk2utt, f_spk2utt_rem)


def GetEgs(datainfo_filepath, out_dir, path_suffix,
           n_ark, negs_per_ark_list, chunklen_range, chunklen_mode='fixed', numofspkrs=None,
           pick_mode='random', segment_hop=None, nspkr_per_eg_list=None, nutt_per_spkr_list=None, spkrmod_per_eg_list=None, ark_remark=None,
           seed=None, out_suffix=None, out_basename=None, **kwargs):
    """ Pick specific n_ark * negs_per_ark examples from spk2utt.pkl """
    """ 
    Args:
        `datainfo_filepath`: [str] filepath to spk2utt.pkl
        `out_filepath`: [str] filepath where out_filename will be saved
        `chunklen_range`: [2-element list] 2-element list: contains minimum and maximum number of frames per archive in ascending order
        `n_ark`: [int] number of archives
        `negs_per_ark_list`: [int] number of examples in each archive
        `kwargs`: [dict] containing 'fs', 'nfft', 'hop'

    Out:
        `all_egs`: [Not returned but saved][list]
                   [{spkr-id-0: [(uttpath, offset, chunklen), (), ...]}, {archive-1}, {archive-2}, ...]
        `out_filepath`: pickle file that saves all_egs
    """

    def get_chunk_len(nframe_per_ark, n_ark, chunklen_mode):
        if chunklen_mode == 'fixed':
            chunklen = int(sum(nframe_per_ark) / len(nframe_per_ark))
            return [chunklen] * n_ark
        elif chunklen_mode == 'random':
            return [random.randint(nframe_per_ark[0], nframe_per_ark[1]) for _ in range(n_ark)]
        elif chunklen_mode == 'linear':
            return [int(round(n)) for n in np.linspace(nframe_per_ark[0], nframe_per_ark[1], num=n_ark)]

    def get_key2segs(spk2utt, segment_len, segment_hop):
        key2segs = defaultdict(list)
        key2seg2list = []
        for key, utts in spk2utt.items():
            total_seg = 0
            for uttid, (_, _, dur) in enumerate(utts):
                # n_seg = int((dur - segment_hop) // segment_len + 1)
                n_seg = int((dur - segment_hop) // segment_len)
                key2segs[key].extend([(uttid, idx) for idx in range(n_seg)])
                total_seg += n_seg
            assert total_seg == len(key2segs[key])
            key2seg2list.append((key, total_seg))
        return key2segs, key2seg2list


    def get_key2chosenegs(n_egs, key2segs, key2seg2list):
        chosennumsegs = [0] * len(key2seg2list)
        keys, totalnumsegs = list(zip(*key2seg2list))
        processed_egs = 0
        while True:
            if processed_egs == n_egs:
                break
            for idx, num in enumerate(chosennumsegs):
                if num < totalnumsegs[idx]:
                    chosennumsegs[idx] += 1
                    processed_egs += 1
                    if processed_egs == n_egs:
                        break
        key2seg2list = list(zip(keys, chosennumsegs))
        key2chosensegs = defaultdict(list)
        for key, numsegs in key2seg2list:
            key2chosensegs[key] = random.sample(key2segs[key], k=numsegs)
        return key2chosensegs

    abs_out_dir = os.path.join(path_suffix, out_dir)
    os.makedirs(abs_out_dir, exist_ok=True)
    if seed is None:
        seed = datetime.datetime.now()
    random.seed(seed)
    ori_basename = os.path.basename(datainfo_filepath)
    ori_basename = ori_basename.split('.', 1)[1].rsplit('.', 1)[0] # 'clean'
    if out_suffix is not None:
        ori_basename = ori_basename + '.' + out_suffix
    if out_basename is None:
        out_basename = '.'.join(['Egs', ori_basename])
        readme_basename = '.'.join(['Readme-GetEgs', ori_basename, 'txt'])
    else:
        readme_basename = 'Readme-' + out_basename + '.txt'

    fs = kwargs['fs']
    stft_hop = int(kwargs['stft_hop'] * fs) #256
    with open(datainfo_filepath, 'rb') as f_spk2utt:
        spk2utt = pickle.load(f_spk2utt)
    key_list = sorted(list(spk2utt.keys()))
    key2realidx = {}
    for idx, key in enumerate(key_list):
        key2realidx[key] = idx
    chunklen_list = get_chunk_len(chunklen_range, n_ark, chunklen_mode)
    random.shuffle(chunklen_list)
    if nspkr_per_eg_list is None:
        nspkr_per_eg_list = [1 for _ in range(n_ark)]
    if numofspkrs is None:
        numofspkrs = len(key_list)

    f_readme = open(os.path.join(abs_out_dir, readme_basename), 'wt')
    print(f"Get examples from {datainfo_filepath}", file=f_readme)
    print(f"Chunk length range in archives: {chunklen_range}", file=f_readme)
    print(f"Chunk length choosing mode: {chunklen_mode}", file=f_readme)
    print(f"Use {numofspkrs} speakers", file=f_readme)
    print(f"Number of archives: {n_ark}", file=f_readme)
    print(f"Number of examples in each archive: {negs_per_ark_list}", file=f_readme)
    print(f"Mode picking examples: {pick_mode}", file=f_readme)
    if pick_mode == 'compact':
        print(f"Segment hop percentage: {segment_hop}", file=f_readme)
    print(f"Number of speakers in each training example: {nspkr_per_eg_list}", file=f_readme)
    print(f"Number of utterances of each speaker in each training example: {nutt_per_spkr_list}", file=f_readme)
    print(f"Speaker mode in each training example: {spkrmod_per_eg_list}", file=f_readme)
    print(f"Sampling rate [Hz]: {fs}", file=f_readme)
    print(f"Hop length for STFT: {stft_hop}", file=f_readme)
    print(f"Out directory: {abs_out_dir}", file=f_readme)
    print(f"Out basename: {out_basename}", file=f_readme)

    egs_cols = ['SpkrID', 'RelativePath', 'Offset', 'Duration', 'Sr', 'SpkrID-original']
    ark_cols = ['RelativePath', 'NumberOfEgsPerArk', 'NumberOfSpeakersPerEgs', 'NumberOfUtterancesPerSpeaker']
    if ark_remark is not None:
        ark_cols.append('Remark')
    ark_info = []
    for ark_id in range(n_ark):
        nspkr_per_eg = nspkr_per_eg_list[ark_id]
        nutt_per_spkr = nutt_per_spkr_list[ark_id]
        spkrmod_per_eg = spkrmod_per_eg_list[ark_id]
        negs_per_ark = negs_per_ark_list[ark_id]
        chunk_len = int((chunklen_list[ark_id] - 1) * stft_hop)
        

        if pick_mode == 'compact':
            egs_in_ark = []
            # TODO: nspkr_per_eg != 1
            segment_hop = int(chunk_len * segment_hop)
            key2segs, key2seg2list = get_key2segs(spk2utt, chunk_len, segment_hop)
            if sum([num for _, num in (key2seg2list)]) < negs_per_ark:
                raise ValueError(f"number of examples with chunk length {chunklen_list[ark_id]} exceeds maximum number of examples avaliable in 'compact' mode!")
            key2chosensegs = get_key2chosenegs(negs_per_ark, key2segs, key2seg2list)
            for key, chosensegs in key2chosensegs.items():
                realid = key2realidx[key] 
                for uttid, local_id in chosensegs:
                    uttpath, startpoint, dur = spk2utt[key][uttid]
                    # offset = startpoint + local_id * segment_hop
                    offset = startpoint + local_id * chunk_len
                    egs_in_ark.append([realid, uttpath, offset, chunk_len, fs, str(key)])
            outname_csv = '.'.join([out_basename,'csv'])
            outpath = os.path.join(abs_out_dir, outname_csv)
            df = pd.DataFrame(egs_in_ark, columns=egs_cols)
            df.to_csv(outpath, index_label='Index')
        elif pick_mode == 'random':
            egs_in_ark = [[] for _ in range(sum(nutt_per_spkr))]
            random.shuffle(key_list)
            key_list = key_list[: numofspkrs]
            processed_egs = 0
            while True:
                if processed_egs == negs_per_ark:
                    break
                for key in key_list:
                    keys_in_eg = []
                    for spkrmod in spkrmod_per_eg:
                        if spkrmod == 'different':
                            key_list_copy = copy.deepcopy(key_list)
                            key_list_copy.remove(key)
                            keys_in_eg.extend(random.sample(key_list_copy, k=1))
                        elif spkrmod == 'same':
                            keys_in_eg.append(key)
                    keys_in_eg.insert(0, key)
                    realindices = [key2realidx[i] for i in keys_in_eg]
                    j = 0
                    for idx in range(nspkr_per_eg):
                        for _ in range(nutt_per_spkr[idx]):
                            uttpath, startpoint, dur = random.choice(spk2utt[keys_in_eg[idx]])  # randomly pick a utterance from the specific spkr
                            assert dur > chunk_len
                            offset = startpoint + random.randint(0, dur - chunk_len - 1)  # randomly pick an offset
                            egs_in_ark[j].append([realindices[idx], uttpath, offset, chunk_len, fs, str(keys_in_eg[idx])])
                            j += 1
                    processed_egs += 1
                    if processed_egs == negs_per_ark:
                        break

            outname = '.'.join([out_basename, str(ark_id), 'xls'])
            outpath = os.path.join(abs_out_dir, outname)
            writer = pd.ExcelWriter(outpath)
            for i in range(sum(nutt_per_spkr)): # nutt_per_spkr=[1,6]
                df = pd.DataFrame(egs_in_ark[i], columns=egs_cols)
                df.to_excel(writer, sheet_name=str(i), index_label='Index')
            writer.save()
            writer.close()
            ark_info.append([os.path.join(out_dir, outname), negs_per_ark, nspkr_per_eg, ':'.join(map(str, nutt_per_spkr))])
            if ark_remark is not None:
                ark_info[-1].append(ark_remark[ark_id])
            arkinfo_outpath = os.path.join(abs_out_dir, '.'.join([out_basename, 'ARKINFO', 'csv']))
            df = pd.DataFrame(ark_info, columns=ark_cols)
            df.to_csv(arkinfo_outpath, index_label='Index')
    f_readme.close()


def main(state=0, sub_state=0, ssd_prefix=None, hdd_prefix=None):
    if state == 0:
        # datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/spk2utt.clean.pkl'
        # out_dir = '/data/hdd0/leleliao/DATASET/LibriSpeech/train/ampnorm_meannorm_trim30'
        # out_namestr = '100spkr-max'
        # max_key_num = 100

        # datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/dev/ampnorm_meannorm_trim30/spk2utt.clean.pkl'
        # out_dir = '/data/hdd0/leleliao/DATASET/LibriSpeech/dev/ampnorm_meannorm_trim30'
        # out_namestr = '40spkr-max'
        # max_key_num = 40

        # datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/test/ampnorm_meannorm_trim30/spk2utt.clean.pkl'
        # out_dir = '/data/hdd0/leleliao/DATASET/LibriSpeech/test/ampnorm_meannorm_trim30'
        # out_namestr = '40spkr-max'
        # max_key_num = 40

        datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/test_dev/ampnorm_meannorm_trim30/spk2utt.clean.pkl'
        out_dir = '/data/hdd0/leleliao/DATASET/LibriSpeech/test_dev/ampnorm_meannorm_trim30'
        out_namestr = '80spkr-max'
        max_key_num = 80


        key_keep_mode = 'max'  # max/min/random
        max_utt_per_key = None
        utt_keep_mode = None
        seed = datetime.datetime.now()

        MakeSubset(datainfo_path, out_dir, out_namestr,
                   max_key_num=max_key_num, key_keep_mode=key_keep_mode,
                   max_utt_per_key=max_utt_per_key, utt_keep_mode=utt_keep_mode,
                   seed=seed)

    elif state == 1:
        """ Split spk2utt.pkl into train, vad, diagnosis """
        # datainfo_path = ['/data/hdd0/leleliao/DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/spk2utt.clean.100spkr-max.pkl']
        # out_path = [os.path.dirname(path) for path in datainfo_path]
        # uttnum_tosplit = [int(400 * 4)]
        # namestr_for_remain = ['train']
        # namestr_for_split = ['dev-test']

        datainfo_path = ['/data/hdd0/leleliao/DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/spk2utt.clean.100spkr-max.dev-test.pkl']
        out_path = [os.path.dirname(path) for path in datainfo_path]
        uttnum_tosplit = [int(400 * 2)]
        namestr_for_remain = ['dev']
        namestr_for_split = ['test']

        exclude = [True]
        min_utt = [8]

        for i in range(len(datainfo_path)):
            SplitDataset(datainfo_path[i], out_path[i], uttnum_tosplit[i], namestr_for_remain[i], namestr_for_split[i],
                         exclude=exclude[i], min_utt=min_utt[i], seed=datetime.datetime.now())

    elif state == 2:
        numofspkrs = 100
        chunklen_range = [200] * 1 
        chunklen_mode = 'fixed' 
        segment_hop = 0 
        seed = datetime.datetime.now()
        fs = 16000
        stft_hop = 0.016
        out_suffix = None


        # out_basename = 'Egs.train.stage1'
        # n_ark = 1
        # pick_mode = 'compact' # 'random' or 'compact'，stage 1的干净数据用 'compact'
        # # 无缝隙地一段接着一段取
        # nspkr_per_eg_list = [2] * n_ark
        # nutt_per_spkr_list = [[1, 6]] * n_ark
        # spkrmod_per_eg_list = [['different']] * n_ark #[['same']] * 1 or [['different']] * 1
        # ark_remark = ['pure'] * 10 #'pure'，'permute'，'mix', or 

        # datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/spk2utt.clean.100spkr-max.train.pkl'
        # out_dir = 'DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/train'
        # negs_per_ark_list = [numofspkrs * 300] * 1 # stage 1

        # datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/spk2utt.clean.100spkr-max.test.pkl'
        # out_dir = 'DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/test'
        # negs_per_ark_list = [numofspkrs * 38] * 1 # stage 1

        # datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/spk2utt.clean.100spkr-max.dev.pkl'
        # out_dir = 'DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/dev'
        # negs_per_ark_list = [numofspkrs * 38] * 1 # stage 1

        # datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/spk2utt.clean.100spkr-max.dev-test.pkl'
        # out_dir = 'DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/dev-test'
        # negs_per_ark_list = [numofspkrs * 76] * 1 # stage 1



        out_basename = 'Egs.train.stage2'
        n_ark = 3
        pick_mode = 'random'
        nspkr_per_eg_list = [2] * n_ark
        nutt_per_spkr_list = [[1, 6]] * n_ark
        spkrmod_per_eg_list = [['different']] * n_ark #[['same']] * 1 or [['different']] * 1
        ark_remark = ['permute','mix','pure'] #'pure'，'permute'，'mix', or None

        # datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/spk2utt.clean.100spkr-max.train.pkl'
        # out_dir = 'DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/train'
        # negs_per_ark_list = [numofspkrs * 200, numofspkrs * 200, numofspkrs * 100] # stage 2 permute or mix

        # datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/spk2utt.clean.100spkr-max.test.pkl'
        # out_dir = 'DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/test'
        # negs_per_ark_list = [numofspkrs * 26, numofspkrs * 26, numofspkrs * 13]
        
        # datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/spk2utt.clean.100spkr-max.dev.pkl'
        # out_dir = 'DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/dev'
        # negs_per_ark_list = [numofspkrs * 26, numofspkrs * 26, numofspkrs * 13]

        # datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/spk2utt.clean.100spkr-max.dev-test.pkl'
        # out_dir = 'DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/dev-test'
        # negs_per_ark_list = [numofspkrs * 50, numofspkrs * 50, numofspkrs * 25]


        # #### test set
        # out_basename = 'Egs.test.stage1'
        # n_ark = 1
        # pick_mode = 'compact' # 'random' or 'compact'，stage 1的干净数据用 'compact'
        # # 无缝隙地一段接着一段取
        # nspkr_per_eg_list = [2] * n_ark
        # nutt_per_spkr_list = [[1, 6]] * n_ark
        # spkrmod_per_eg_list = [['different']] * n_ark #[['same']] * 1 or [['different']] * 1
        # ark_remark = ['pure'] * 10 #'pure'，'permute'，'mix', or 

        # datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/test/ampnorm_meannorm_trim30/spk2utt.clean.40spkr-max.pkl'
        # out_dir = 'DATASET/LibriSpeech/test/ampnorm_meannorm_trim30/segments'
        # numofspkrs = 40
        # negs_per_ark_list = [numofspkrs * 100] * 1 # stage 1

        # #### dev set
        # out_basename = 'Egs.dev.stage1'
        # n_ark = 1
        # pick_mode = 'compact' # 'random' or 'compact'，stage 1的干净数据用 'compact'
        # # 无缝隙地一段接着一段取
        # nspkr_per_eg_list = [2] * n_ark
        # nutt_per_spkr_list = [[1, 6]] * n_ark
        # spkrmod_per_eg_list = [['different']] * n_ark #[['same']] * 1 or [['different']] * 1
        # ark_remark = ['pure'] * 10 #'pure'，'permute'，'mix', or 

        # datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/dev/ampnorm_meannorm_trim30/spk2utt.clean.40spkr-max.pkl'
        # out_dir = 'DATASET/LibriSpeech/dev/ampnorm_meannorm_trim30/segments'
        # numofspkrs = 40
        # negs_per_ark_list = [numofspkrs * 100] * 1 # stage 1


        # #### test-dev set
        # out_basename = 'Egs.test-dev.stage1'
        # n_ark = 1
        # pick_mode = 'compact' # 'random' or 'compact'，stage 1的干净数据用 'compact'
        # # 无缝隙地一段接着一段取
        # nspkr_per_eg_list = [2] * n_ark
        # nutt_per_spkr_list = [[1, 6]] * n_ark
        # spkrmod_per_eg_list = [['different']] * n_ark #[['same']] * 1 or [['different']] * 1
        # ark_remark = ['pure'] * 10 #'pure'，'permute'，'mix', or 

        # datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/test_dev/ampnorm_meannorm_trim30/spk2utt.clean.80spkr-max.pkl'
        # out_dir = 'DATASET/LibriSpeech/test_dev/ampnorm_meannorm_trim30/segments'
        # numofspkrs = 80
        # negs_per_ark_list = [numofspkrs * 100] * 1 # stage 1


        # #### test set
        # out_basename = 'Egs.test.stage2'
        # n_ark = 3
        # pick_mode = 'random'
        # nspkr_per_eg_list = [2] * n_ark
        # nutt_per_spkr_list = [[1, 6]] * n_ark
        # spkrmod_per_eg_list = [['different']] * n_ark #[['same']] * 1 or [['different']] * 1
        # ark_remark = ['permute','mix','pure'] #'pure'，'permute'，'mix', or None

        # datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/test/ampnorm_meannorm_trim30/spk2utt.clean.40spkr-max.pkl'
        # out_dir = 'DATASET/LibriSpeech/test/ampnorm_meannorm_trim30/segments'
        # numofspkrs = 40
        # negs_per_ark_list = [numofspkrs * 70, numofspkrs * 70, numofspkrs * 35] # stage 2 permute or mix


        # #### dev set
        # out_basename = 'Egs.dev.stage2'
        # n_ark = 3
        # pick_mode = 'random'
        # nspkr_per_eg_list = [2] * n_ark
        # nutt_per_spkr_list = [[1, 6]] * n_ark
        # spkrmod_per_eg_list = [['different']] * n_ark #[['same']] * 1 or [['different']] * 1
        # ark_remark = ['permute','mix','pure'] #'pure'，'permute'，'mix', or None

        # datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/dev/ampnorm_meannorm_trim30/spk2utt.clean.40spkr-max.pkl'
        # out_dir = 'DATASET/LibriSpeech/dev/ampnorm_meannorm_trim30/segments'
        # numofspkrs = 40
        # negs_per_ark_list = [numofspkrs * 70, numofspkrs * 70, numofspkrs * 35] # stage 2 permute or mix


        #### test-dev set
        out_basename = 'Egs.test-dev.stage2'
        n_ark = 3
        pick_mode = 'random'
        nspkr_per_eg_list = [2] * n_ark
        nutt_per_spkr_list = [[1, 6]] * n_ark
        spkrmod_per_eg_list = [['different']] * n_ark #[['same']] * 1 or [['different']] * 1
        ark_remark = ['permute','mix','pure'] #'pure'，'permute'，'mix', or None

        datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/test_dev/ampnorm_meannorm_trim30/spk2utt.clean.80spkr-max.pkl'
        out_dir = 'DATASET/LibriSpeech/test_dev/ampnorm_meannorm_trim30/segments'
        numofspkrs = 80
        negs_per_ark_list = [numofspkrs * 70, numofspkrs * 70, numofspkrs * 35] # stage 2 permute or mix


        GetEgs(datainfo_path, out_dir, hdd_prefix,
               n_ark, negs_per_ark_list, chunklen_range, chunklen_mode=chunklen_mode, numofspkrs=numofspkrs,
               pick_mode=pick_mode, segment_hop=segment_hop, nspkr_per_eg_list=nspkr_per_eg_list, nutt_per_spkr_list=nutt_per_spkr_list,
               spkrmod_per_eg_list=spkrmod_per_eg_list, ark_remark=ark_remark,
               seed=seed, fs=fs, stft_hop=stft_hop,
               out_suffix=out_suffix, out_basename=out_basename)


if __name__ == "__main__":
    hdd_prefix='/data/hdd0/leleliao'
    main(state=2, sub_state=0, hdd_prefix=hdd_prefix)
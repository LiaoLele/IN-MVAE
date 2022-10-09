import os
import random
import pickle
import datetime
import pandas as pd
from utils import get_pair2utts
from itertools import product
from collections import defaultdict


def MakePair(srcinfo_path, out_dir, src_num=2, mode='print_avaliable_pair_number',
             n_pair=None, n_egs_per_pair=None,
             subset_func=None, subset_func_args=None,
             fs=16000, cols=None,
             seed=None, suffix=''):
             
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

    if seed is None:
        seed = datetime.datetime.now()
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    out_basename = os.path.basename(srcinfo_path).split('.')[1: -1] #spk2utt.clean.100spkr-max.test.pkl
    f_readme = open(os.path.join(out_dir, '.'.join(['Readme-MakePair'] + out_basename + [suffix, 'txt'])), 'wt')
    #Readme-MakePair.clean.100spkr-max.test.2src-FM-40*1.txt
    with open(srcinfo_path, 'rb') as f:
        spk2utt = pickle.load(f)
    if subset_func is not None:
        spk2utt = subset_func(spk2utt, **subset_func_args)
    else:
        spk2utt = {'Not specified': spk2utt}
    if len(list(spk2utt.keys())) == 1:
        subset_name = [list(spk2utt.keys())[0]] * src_num
        subset_data = [{} for _ in range(src_num)]
        keylist = list(spk2utt[subset_name[0]].keys())
        random.shuffle(keylist)
        for i, key in enumerate(keylist):
            subset_data[i % src_num][key] = spk2utt[subset_name[0]][key]
    else:
        subset_name = [list(spk2utt.keys())[1], list(spk2utt.keys())[1]]
        subset_data = list(spk2utt.values())[1]
        src_num = 2
    print("The sources are from {}".format(srcinfo_path), file=f_readme)
    print("Number of sources in each pair: {}".format(src_num), file=f_readme)
    print("Number of speaker pairs: {}".format(n_pair), file=f_readme)
    print("Number of utterances per speaker pair: {}".format(n_egs_per_pair), file=f_readme)
    print("Whether to use all speakers: True", file=f_readme)
    mix2pair = [[] for _ in range(src_num)]

    # use all keys to create n_pair
    pair_list = []
    subset_data_keylist = list(subset_data.keys())
    subset_data_keynum = len(subset_data_keylist)

    pair_list = list(zip([subset_data_keylist[0] for _ in range(subset_data_keynum - 1)], subset_data_keylist[1:-1]))
    random.shuffle(pair_list)
    if len(pair_list) >= n_pair:
        pair_list = pair_list[: n_pair]
    else:
        raise ValueError(f"Required number of pairs {n_pair} exceeds total number of female speaker! Please reconsider!")

    # choose utterance
    pair2utts, pair2totaluttnum = get_pair2utts(pair_list, [subset_data, subset_data])
    required_uttnum = int(n_pair * n_egs_per_pair)
    if sum(list(pair2totaluttnum.values())) < required_uttnum:
        raise ValueError(f"Required number of utterance {required_uttnum} exceeds total number of utterance available {sum(pair2totaluttnum)}! Please reconsider!")
    pair2uttnum = defaultdict(int)
    num_generated_utts = 0
    while num_generated_utts < required_uttnum:
        for spkrs in pair2totaluttnum.keys():
            if pair2uttnum[spkrs] < pair2totaluttnum[spkrs]:
                pair2uttnum[spkrs] += 1
                num_generated_utts += 1
                if num_generated_utts >= required_uttnum:
                    break
    for spkrs, uttids in pair2utts.items():
        uttnum = pair2uttnum[spkrs]
        chosen_uttids = random.sample(uttids, uttnum)
        for uttid in chosen_uttids:
            for i in range(src_num):
                mix2pair[i].append(list([subset_data, subset_data][i][spkrs[i]][uttid[i]]) + [fs] + [subset_name[i]])
    out_name = os.path.join(out_dir, '.'.join(['Pair'] + out_basename + [suffix, 'xls']))
    print(f"Pair information saved in {out_name}", file=f_readme)
    writer = pd.ExcelWriter(out_name)
    for i in range(src_num):
        df = pd.DataFrame(mix2pair[i], columns=cols)
        df.to_excel(writer, sheet_name=str(i), index_label='Index')
    writer.save()
    writer.close()


def ModRIR(ririnfo_path, pair_path, usecols=None, assign_key=None, mode=0, suffix='', del_orifile=False):

    out_path = pair_path.rsplit('.', 1)
    out_path.insert(1, suffix)
    out_path = '.'.join(out_path)
    excel_reader = pd.ExcelFile(pair_path)
    sheet_name_list = excel_reader.sheet_names
    pair_num = excel_reader.parse(sheet_name='0', header=0).shape[0]
    print('In total, there are {} pairs to be mixed.'.format(pair_num))
    with open(ririnfo_path, 'rb') as f:
        key2gidx = pickle.load(f)
    rir_list = []

    if assign_key is None:
        pair_idx = 0
        while True:
            for key in key2gidx.keys():
                if pair_idx == pair_num:
                    break
                rir_idx = random.choice(key2gidx[key])
                rir_list.append(rir_idx)
                pair_idx += 1
            if pair_idx == pair_num:
                break
    else:
        if mode == 0:
            """ key2gidx: {"0.15-20": [idx-0, idx-1,...], ...} """
            """ for test_simu """
            for pair_idx in range(pair_num):
                rir_idx = random.choice(key2gidx[assign_key])
                rir_list.append(rir_idx)
        elif mode == 1:
            """ key2gidx: {"0.160": [[angle-interval=20], [angle-interval=30], ...]} """
            """ for test_real_2_src """
            pair_idx = 0
            while True:
                for rir_idx_list in key2gidx[assign_key]:
                    if pair_idx == pair_num:
                        break
                    rir_idx = random.choice(rir_idx_list)
                    rir_list.append(rir_idx)
                    pair_idx += 1
                if pair_idx == pair_num:
                    break
    with pd.ExcelWriter(out_path) as writer:
        for sheet_name in sheet_name_list:
            if sheet_name not in ['SIR', 'RIRidx']:
                df = excel_reader.parse(sheet_name=sheet_name, header=0, usecols=usecols)
                df.to_excel(writer, sheet_name=sheet_name, index_label="Index")
            elif sheet_name == 'SIR':
                df = excel_reader.parse(sheet_name=sheet_name, header=0, usecols=['SIR'])
                df.to_excel(writer, sheet_name=sheet_name, index_label="Index")
        df = pd.DataFrame({'RIRidx': rir_list})
        df.to_excel(writer, sheet_name='RIRidx', index_label='Index')
    if del_orifile:
        os.system(f"rm {pair_path}")
    

def ModSIR(sir_range, pair_path, usecols=None, use_random=False):

    excel_reader = pd.ExcelFile(pair_path)
    sheet_name_list = excel_reader.sheet_names
    pair_num = excel_reader.parse(sheet_name='0', header=0).shape[0]
    print('In total, there are {} pairs to be mixed.'.format(pair_num))
    sir_list = []

    if use_random:
        for idx in range(pair_num):
            sir_list.append(random.choice(sir_range))
    else:
        pair_idx = 0
        while True:
            for sir in sir_range:
                if pair_idx == pair_num:
                    break
                sir_list.append(sir)
                pair_idx += 1
            if pair_idx == pair_num:
                break
    with pd.ExcelWriter(pair_path) as writer:
        for sheet_name in sheet_name_list:
            if sheet_name not in ['SIR', 'RIRidx']:
                df = excel_reader.parse(sheet_name=sheet_name, header=0, usecols=usecols)
                df.to_excel(writer, sheet_name=sheet_name, index_label="Index")
            elif sheet_name == 'RIRidx':
                df = excel_reader.parse(sheet_name=sheet_name, header=0, usecols=['RIRidx'])
                df.to_excel(writer, sheet_name=sheet_name, index_label="Index")
        df = pd.DataFrame({'SIR': sir_list})
        df.to_excel(writer, sheet_name='SIR', index_label='Index')


def main(state=0, sub_state=0, ssd_prefix=None, hdd_prefix=None):
    if state == 0:
        def subset_func(spk2utt, sexinfo=None, sheet_name_list=None, usecols=None):
            spk2utt_tmp = {}
            for key in spk2utt.keys():
                real_key = int(spk2utt[key][0][0].rsplit('/', 1)[1].split('-')[1].split('.')[0])
                spk2utt_tmp[real_key] = spk2utt[key]
            spk2utt = spk2utt_tmp
            excel_reader = pd.ExcelFile(sexinfo)
            classes = defaultdict(list)
            for sheet_name in sheet_name_list:
                df = excel_reader.parse(sheet_name=sheet_name, header=0, usecols=usecols)
                classes['Female'].extend(list(df['Female'].dropna()))
                classes['Male'].extend(list(df['Male'].dropna()))
            spk2utt_new = {}
            for clas in classes.keys():
                spk2utt_new[clas] = {}
                target_key = list(filter(lambda x: x in classes[clas], list(spk2utt.keys())))
                for key in target_key:
                    spk2utt_new[clas][key] = spk2utt[key]
            return spk2utt_new
        subset_func_args = {
            'sexinfo': os.path.join(ssd_prefix, 'DATASET/LibriSpeech/SexualInfo.xls'),
            'sheet_name_list': ['train-clean-100', 'train-clean-360'],
            'usecols': ['Female', 'Male'],
        }

        srcinfo_path = os.path.join(hdd_prefix, 'DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/spk2utt.clean.100spkr-max.test.pkl')
        # srcinfo_path = os.path.join(hdd_prefix, 'PROJECT/CVAE_training/EsEc_structure/data/inference/spk2utt.5s.train.pkl')
        out_dir = os.path.join(hdd_prefix, 'DATASET/LibriSpeech/inference')
        src_num = 2
        mode = 'generate'
        n_pair = 40
        n_egs_per_pair = 1
        fs = 16000
        cols = ['RelativePath', 'Offset', 'Duration', 'Sr', 'Subset']
        # seed = datetime.datetime.now()
        seed = 3
        suffix = '2src-MM-40_1'

        MakePair(srcinfo_path, out_dir, src_num=src_num, mode=mode,
                 n_pair=n_pair, n_egs_per_pair=n_egs_per_pair,
                 subset_func=subset_func, subset_func_args=subset_func_args,
                 fs=fs, cols=cols,
                 seed=seed, suffix=suffix)

    if state == 1:
        ririnfo_path = os.path.join(hdd_prefix, 'DATASET/rir/MIRD/key2idx.pkl')
        pair_path = os.path.join(hdd_prefix, 'DATASET/LibriSpeech/inference/Pair.clean.100spkr-max.test.2src-MM-40_1.xls')
        assign_key = '0.360'
        mode = 1
        usecols = ['RelativePath', 'Offset', 'Duration', 'Sr', 'Subset']
        suffix = 't60-36ms'
        del_orifile = False

        ModRIR(ririnfo_path, pair_path, usecols=usecols, assign_key=assign_key, mode=mode, suffix=suffix, del_orifile=del_orifile)
    
    if state == 2:
        sir_range = [-5, 0, 5]
        pair_path = os.path.join(hdd_prefix, 'DATASET/LibriSpeech/inference/Pair.clean.100spkr-max.test.2src-MM-40_1.t60-36ms.xls')
        usecols = ['RelativePath', 'Offset', 'Duration', 'Sr', 'Subset']
        use_random = False
        ModSIR(sir_range, pair_path, usecols=usecols, use_random=use_random)

if __name__ == "__main__":
    hdd_prefix = '/data/hdd0/leleliao/'
    ssd_prefix = '/data/hdd0/leleliao/'
    main(state=2, sub_state=0, ssd_prefix=ssd_prefix, hdd_prefix=hdd_prefix)
    
import  matplotlib.pyplot as plt
from itertools import cycle
from scipy import stats
import pandas as pd
import numpy as np
import datetime
import random
import os


def CreatePermuteData(xls_path, split_rules, total_point, usecols=None, seed=None, out_basename=None):

    if seed is None:
        seed = datetime.datetime.now()
    random.seed(seed)
    excel_reader = pd.ExcelFile(xls_path)
    sheet_name_list = list(excel_reader.sheet_names) 
    # range(sum(nutt_per_spkr))：nutt_per_spkr=[1,6]
    # sheet_name_list = range(7)
    if 'SplitInfo' in sheet_name_list:
        sheet_name_list.remove('SplitInfo')
    pair_num = excel_reader.parse(sheet_name='0', header=0).shape[0] # negs_per_ark
    print('In total, there are {} examples.'.format(pair_num))

    split_rules_list = split_rules.split('@')
    split_rule_gen = cycle(split_rules_list)
    split_info = []
    for idx in range(pair_num):
        num_block, start_pt, minlen_perblock, mindis_betweenblock = map(int, next(split_rule_gen).split(":"))
        blocklen_list = []
        intervallen_list = []
        for block_id in range(num_block):
            avglen_perblock = int((total_point - start_pt - (num_block - 1) * mindis_betweenblock - sum(blocklen_list)) / (num_block - block_id))
            assert avglen_perblock >= minlen_perblock
            blocklen_list.append(random.randint(minlen_perblock, avglen_perblock))
        for interval_id in range(num_block - 1):
            avglen_perinterval = int((total_point - start_pt - sum(blocklen_list) - sum(intervallen_list)) / (num_block - 1 - interval_id))
            assert avglen_perinterval >= mindis_betweenblock
            intervallen_list.append(random.randint(mindis_betweenblock, avglen_perinterval))
        rest_len = int(total_point - start_pt - sum(blocklen_list) - sum(intervallen_list))
        start_margin = random.randint(0, rest_len)
        random.shuffle(blocklen_list)
        random.shuffle(intervallen_list)
        intervallen_list.append(int(rest_len - start_margin))
        pt = start_pt + start_margin
        split_eg = []
        for i in range(num_block):
            split_eg.append(slice(pt, pt + blocklen_list[i]))
            pt = pt + blocklen_list[i] + intervallen_list[i]
        assert pt == total_point
        split_info.append(str(split_eg))

    if out_basename is None:
        out_path = xls_path
    else:
        out_path = os.path.join(os.path.dirname(xls_path), out_basename)
    with pd.ExcelWriter(out_path) as writer:
        for sheet_name in sheet_name_list:
            df = excel_reader.parse(sheet_name=sheet_name, header=0, usecols=usecols)
            df.to_excel(writer, sheet_name=sheet_name, index_label="Index")
        df = pd.DataFrame({'SplitRange': split_info})
        df.to_excel(writer, sheet_name='SplitInfo', index_label='Index')


def PlotPDF():
    ncomponent = 10
    mixture_weight_list = random.choices(np.linspace(0.1, 0.5, num=5, endpoint=True), k=ncomponent)
    mixture_weight_list = mixture_weight_list / sum(mixture_weight_list)
    x_range = np.linspace(0, 20, num=513, endpoint=True)
    loc_list = random.sample(list(x_range[129:]), k=ncomponent)
    cum_weights = [30, 40, 50, 60, 65, 70, 75, 80, 85]
    df_list = [1] + random.choices(range(1, 10), cum_weights=cum_weights, k=ncomponent - 1)
    peak = np.random.uniform(low=0.5, high=0.9, size=(1))[0]
    y = []
    for i in range(ncomponent):
        df = df_list[i]
        loc = loc_list[i]
        mixture_weight = mixture_weight_list[i]
        y.append(mixture_weight * stats.t.pdf(x_range, df, loc))
    y = sum(y)
    y = y / np.max(y) * peak
    plt.plot(np.linspace(0, 513, num=513, endpoint=False), y)
    plt.show()
    print(mixture_weight_list)
    print(loc_list)


def main(state=0, sub_state=0, hdd_prefix=None, ssd_prefix=None):

    if state == 0:
        # xls_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/train/Egs.train.stage2.0.xls'
        # xls_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/test/Egs.train.stage2.0.xls'
        # xls_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/dev/Egs.train.stage2.0.xls'
        # xls_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/dev-test/Egs.train.stage2.0.xls'
        # xls_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/test/ampnorm_meannorm_trim30/segments/Egs.test.stage2.0.xls'
        # xls_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/dev/ampnorm_meannorm_trim30/segments/Egs.dev.stage2.0.xls'
        xls_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/test_dev/ampnorm_meannorm_trim30/segments/Egs.test-dev.stage2.0.xls'
        out_basename = None
        out_basename = None
        split_rules = "1:128:64:0@2:128:64:64@3:128:50:64"  
        # numofblock:startpoint:minlenofeachblock:mindistancebetweenblocks
        # numofblock:有numofblock个区间有排序问题
        # startpoint:从第startpoint个频点开始产生排序问题的区间
        total_point = 512
        usecols = ['SpkrID', 'RelativePath', 'Offset', 'Duration', 'Sr', 'SpkrID-original']
        # usecols = ['RelativePath', 'Offset', 'Duration', 'Sr', 'Subset']
        CreatePermuteData(xls_path, split_rules, total_point, usecols=usecols, out_basename=out_basename)

    if state == 1:
        PlotPDF()


if __name__ == "__main__":
    main(state=0, sub_state=0)

    

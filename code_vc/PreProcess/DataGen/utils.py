from collections import defaultdict
from itertools import product


def get_key2utts(spk2utt):
    """ 
    Args:
        spk2utt: [dict]
    Out:
        num: [returned][int]
             number of utterances in spk2utt
    """
    key2utts = defaultdict(int)

    for key, value in spk2utt.items():
        key2utts[key] += len(value)
    return key2utts


def get_key2lens(spk2utt):
    key2lens = defaultdict(int)
    for key, value in spk2utt.items():
        for _, _, dur in value:
            key2lens[key] += dur
    return key2lens


def get_pair2utts(pair, subset_data):

    pair2utts = {}
    pair2totaluttnum = {}
    for spkrs in pair:
        src_num = len(spkrs)
        utt_pair = list(product(*[range(len(subset_data[i][spkrs[i]])) for i in range(src_num)]))
        pair2utts[spkrs] = utt_pair
        pair2totaluttnum[spkrs] = len(utt_pair)
    return pair2utts, pair2totaluttnum


    

import re
import pickle
import os
import torch
import soundfile as sf
from torch.utils import data
import copy
import random


class Dataset(data.Dataset):

    """ 
    Args:
        `prefix`: prefix of path when operating on different servers
        `all_egs`: examples to train and test
        `negs_per_spkr`: number of speakers for each training batch

    Classmethod:
        `from_pkl`:
            `datainfo_path`: path where all_egs.pkl is saved
            `prefix` and `negs_per_spkr`: same as in args
    """

    def __init__(self, all_egs, prefix):
        super(Dataset, self).__init__()
        self.all_egs = all_egs
        self.prefix = prefix
        self.ptn = re.compile(r'^.*DATASET/(?P<rel_path>.*)$')

        self.coder()  # create encoder and decoder
        
    @classmethod
    def from_pkl(cls, datainfo_path, prefix):
        with open(datainfo_path, 'rb') as f_all_egs:
            all_egs = pickle.load(f_all_egs)
        return cls(all_egs, prefix)

    @property
    def spkr_num(self):
        spkr_set = set()
        for egs in self.all_egs:
            for spkr in egs.keys():
                spkr_set.add(spkr)
        return len(spkr_set)

    def coder(self):
        """ encoder: encode each utterance with a unique id
            decoder: decode each utterances id to (ark_id, spkr_id, relative_utt_id)
            self.idx2ark: {utt-id-0: (ark-id, spkr-id, rel-utt-id), ...}
            self.ark2idx: [[uttid-for-ark-0], [uttid-for-ark-1], ...] """

        self.idx2ark = {}
        self.ark2idx = [[] for _ in range(len(self.all_egs))]
        utt_id = 0 
        for ark_id, ark in enumerate(self.all_egs):
            for spkr_id, value in ark.items():
                for rel_utt_id in range(len(value)):
                    self.idx2ark[utt_id] = (ark_id, spkr_id, rel_utt_id)
                    self.ark2idx[ark_id].append(utt_id)
                    utt_id += 1
        self.len = utt_id

    def __getitem__(self, idx):
        ark_id, spkr_id, rel_utt_id = self.idx2ark[idx]
        uttpath, offset, chunklen = self.all_egs[ark_id][spkr_id][rel_utt_id]
        uttpath = os.path.join(self.prefix, uttpath) 
        s, _ = sf.read(uttpath, start=offset, stop=offset + chunklen)
        utts = torch.from_numpy(s)
        return utts, torch.tensor(spkr_id) 

    def __len__(self):
        return self.len


class TrainBatchSampler(data.Sampler):

    def __init__(self, ark2idx, batch_size, n_batch=None, drop_last=True):
        self.ark2idx = ark2idx
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.n_batch = self.get_len(n_batch)

    @classmethod
    def from_dataset(cls, dataset, batch_size, n_batch=None, drop_last=True):
        ark2idx = copy.deepcopy(dataset.ark2idx)
        return cls(ark2idx, batch_size, n_batch=n_batch, drop_last=drop_last)

    def get_len(self, n_batch):
        if n_batch is None:
            n_batch = 0
            for ark_idx in self.ark2idx:
                if self.drop_last:
                    n_batch += int(len(ark_idx) // self.batch_size)
                else:
                    n_batch = n_batch + int(len(ark_idx) // self.batch_size) + int(len(ark_idx) % self.batch_size > 0)
        return n_batch

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        batch_in_list = []
        ark2idx_copy = copy.deepcopy(self.ark2idx)
        for ark_idx in ark2idx_copy:
            random.shuffle(ark_idx)
            for i in range(int(len(ark_idx) // self.batch_size)):
                batch_in_list.append(ark_idx[i * self.batch_size: i * self.batch_size + self.batch_size])
            if (not self.drop_last) and (len(ark_idx) % self.batch_size > 0):
                batch_in_list.append(ark_idx[(i + 1) * self.batch_size:])
        random.shuffle(batch_in_list)
        for batch in batch_in_list[0: self.n_batch]:
            yield batch
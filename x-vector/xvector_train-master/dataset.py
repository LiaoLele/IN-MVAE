import torch.utils.data as data
import random
import torch
import os
import numpy as np
import soundfile as sf
import pickle
import re
import time
import copy


""" version 2 """
class Dataset(data.Dataset):

    """ 
    读一段语音信号
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
        # self.ptn = re.compile(r'^.*DATASET/(?P<rel_path>.*)$')

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
        """ An utterance is well determined by ark_id(which arkfile), spkr_id(which speaker) and rel_utt_id(which utt of this speaker) """

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
        # if self.ptn.match(uttpath)['rel_path'].startswith('Librispeech'):
        #     # uttpath = os.path.join('/home/user/zhaoyi.gu/DATA/ori/ori', self.ptn.match(uttpath)['rel_path'].rsplit('/', maxsplit=1)[1])
        #     uttpath = os.path.join('Librispeech_for_proj/ori_data', self.ptn.match(uttpath)['rel_path'].rsplit('/', maxsplit=1)[1])
        # else:
        #     # uttpath = os.path.join('/home/user/zhaoyi.gu/DATA/sep',
        #                         #    self.ptn.match(uttpath)['rel_path'].rsplit('/', maxsplit=2)[1],
        #                         #    self.ptn.match(uttpath)['rel_path'].rsplit('/', maxsplit=2)[2])
        #     uttpath = os.path.join('Librispeech_for_proj/sep',
        #                            self.ptn.match(uttpath)['rel_path'].rsplit('/', maxsplit=2)[1],
        #                            self.ptn.match(uttpath)['rel_path'].rsplit('/', maxsplit=2)[2])
        # uttpath = os.path.join(self.prefix, 'DATASET', self.ptn.match(uttpath)['rel_path'])

        # uttpath = os.path.join(self.prefix, uttpath)
        s, _ = sf.read(uttpath, start=offset, stop=offset + chunklen)
        utts = torch.from_numpy(s)
        return utts, torch.tensor(spkr_id), (uttpath, offset, chunklen)

    def __len__(self):
        return self.len


class TrainBatchSampler(data.Sampler):

    def __init__(self, ark2idx, batch_size, n_batch=None, drop_last=True):
        """ 
        Args:
            `n_batch`: number of batches
        """
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


if __name__ == '__main__':
    """ test batchsampler """
    abs_idx_for_arks_dummy = [list(range(26667)) for _ in range(9)]
    sampler = TrainBatchSampler(abs_idx_for_arks_dummy, 160, drop_last=False)
    t = time.time()
    ss = iter(sampler)
    d = next(ss)
    t = time.time() - t
    print(t)
    # print(len(sampler))
    # for batch in iter(sampler):
    #     print(batch)
    # print('another')
    # for batch in iter(sampler):
    #     print(batch)


""" version 1 """
# class my_dataset(data.Dataset):

#     def __init__(self, data_dir, M, start_time, set_time, spkrs_percet, use_random_len=True):
#         super(my_dataset, self).__init__()

#         self.data_dir = data_dir
#         self.use_random_len = use_random_len
#         f = open(os.path.join(data_dir, 'info.pkl'), 'rb')
#         self.files = pickle.load(f)
#         self.M = M
#         self.start_point = int(60 * start_time * 16000)

#         self.speaker_num = len(self.files)
#         self.set_num = set_time
#         self.spkrs_percent = spkrs_percet

#         self.stft_len = int(hp.data.stft_frame * hp.data.sr)
#         self.stft_hop = int(hp.data.stft_hop * hp.data.sr)

#     def __len__(self):
#         return int(self.spkrs_percent * self.speaker_num)

#     def __getitem__(self, index):
        
#         selected_utter = []
#         if self.use_random_len and self.cnt % hp.train.N == 0:
#             pass
#             # TODO
#         elif self.use_random_len is False:
#             self.frame_len = hp.train.average_frame_num  # frame
#             self.frame_len = int((self.frame_len - 1) * self.stft_hop)  # sample
#             self.frame_hop = int(self.frame_len * hp.train.frame_hop)  # sample
        
#         max_num = int((self.files[index][1] - self.start_point - self.frame_len) // self.frame_hop + 1)
#         sample_num = int(min(self.set_num, max_num))
#         assert self.M <= sample_num
#         select_idx = random.sample(range(sample_num), self.M)

#         for idx in select_idx:   #
#             utter, _ = sf.read(os.path.join(self.data_dir, self.files[index][0].rsplit('/', maxsplit=1)[1]), 
#                                start=self.start_point + idx * self.frame_hop, stop=self.start_point + idx * self.frame_hop + self.frame_len)
#             selected_utter.append(utter)
        
#         selected_utter = torch.from_numpy(np.stack(selected_utter, axis=0))
        
#         return selected_utter, torch.full((self.M, ), self.files[index][2], dtype=int)


# class tmp_dataset(data.Dataset):

#     def __init__(self, data_dir, M, start_time, spkrs_percet, use_random_len=True):
#         super(tmp_dataset, self).__init__()

#         self.data_dir = data_dir
#         self.use_random_len = use_random_len
#         f = open(os.path.join(data_dir, 'info.pkl'), 'rb')
#         self.files_ori = pickle.load(f)
#         self.speaker_num = len(self.files_ori)
#         self.spkrs_percent = spkrs_percet
#         self.len = int(self.spkrs_percent * self.speaker_num)

#         self.files = random.sample(self.files_ori, self.len)

#         self.M = M
#         self.start_point = 0
#         self.end_point = int(2 * 60 * hp.data.sr)        

#         self.stft_len = int(hp.data.stft_frame * hp.data.sr)
#         self.stft_hop = int(hp.data.stft_hop * hp.data.sr)

#     def __len__(self):
#         return self.len
        
#     def __getitem__(self, index):
        
#         selected_utter = []
#         if self.use_random_len and self.cnt % hp.train.N == 0:
#             pass
#             # TODO
#         elif self.use_random_len is False:
#             self.frame_len = hp.train.average_frame_num  # frame
#             self.frame_len = int((self.frame_len - 1) * self.stft_hop)  # sample
            
#         select_idx = random.sample(range(self.end_point - self.frame_len), self.M)

#         for idx in select_idx:   #
#             utter, _ = sf.read(os.path.join(self.data_dir, self.files[index][0]), 
#                                start=self.start_point + idx, stop=self.start_point + idx + self.frame_len)
#             selected_utter.append(utter)
        
#         selected_utter = torch.from_numpy(np.stack(selected_utter, axis=0))
        
#         return selected_utter, torch.full((self.M, ), self.files[index][2], dtype=int)

#     def reshuffle(self):
#         self.files = random.sample(self.files_ori, self.len)


# class my_dataset_test(data.Dataset):

#     def __init__(self, data_dir, enroll_num, verify_num, frame_num=160, enroll_hop=0.5, verify_hop=0.8, num_speaker=None):

#         super(my_dataset_test, self).__init__()
#         self.data_dir = data_dir
#         self.enroll_num = enroll_num
#         self.verify_num = verify_num
#         self.frame_num = frame_num

#         f = open(os.path.join(data_dir, 'info.pkl'), 'rb')
#         self.files = pickle.load(f)
#         if num_speaker is not None:
#             self.files = random.sample(self.files, num_speaker)
#         self.speaker_num = len(self.files)
#         self.frame_num = int((self.frame_num - 1) * hp.data.stft_hop * hp.data.sr)
#         self.enroll_hop = int(self.frame_num * enroll_hop)
#         self.verify_hop = int(self.frame_num * verify_hop)

#         self.frame_len_train = hp.test.average_frame_num_train + 20  # frame
#         self.frame_len_train = int((self.frame_len_train - 1) * hp.data.stft_hop * hp.data.sr)  # sample
#         self.frame_hop_train = int(self.frame_len_train * hp.train.frame_hop)  # sample
#         self.start_point0 = (hp.test.num_utter_per_speaker - 1) * self.frame_hop_train + self.frame_len_train

#     def __len__(self):
#         return self.speaker_num

#     def __getitem__(self, index):

#         selected_utter = []
#         file_len = self.files[index][1] - self.start_point0
#         self.start_point1 = file_len // 2 + self.start_point0
#         enroll_num = int((file_len // 2 - self.frame_num) // self.enroll_hop + 1)
#         verify_num = int((file_len // 2 - self.frame_num) // self.verify_hop + 1)
#         assert self.enroll_num <= enroll_num
#         assert self.verify_num <= verify_num
#         enroll_idx = random.sample(range(enroll_num), self.enroll_num)
#         verify_idx = random.sample(range(verify_num), self.verify_num)
#         for idx in enroll_idx:
#             utter, _ = sf.read(os.path.join(self.data_dir, self.files[index][0]), 
#                                start=self.start_point0 + idx * self.enroll_hop, stop=self.start_point0 + idx * self.enroll_hop + self.frame_num)
#             selected_utter.append(utter)
#         for idx in verify_idx:
#             utter, _ = sf.read(os.path.join(self.data_dir, self.files[index][0]), 
#                                start=self.start_point1 + idx * self.verify_hop, stop=self.start_point1 + idx * self.verify_hop + self.frame_num)
#             selected_utter.append(utter)            

#         selected_utter = np.stack(selected_utter, axis=0)
#         return selected_utter

""" version 0 """
# class my_dataset(data.Dataset):
#     def __init__(self, data_dir, transform, M, mode='spectrogram', use_random_len=True):
#         super(my_dataset, self).__init__()
#         self.data_dir = data_dir
#         self.transform = transform
#         self.M = M
#         self.mode = mode
#         self.use_random_len = use_random_len
#         self.files = list(filter(lambda x: fnmatch.fnmatch(x, 's*.npy'), os.listdir(self.data_dir)))
#         self.speaker_num = len(self.files)
#         if use_random_len:
#             self.cnt = 0
#         self.set_len = hp.data.num_utter_per_speaker

#         self.stft_len = int(hp.data.stft_frame * hp.data.sr)
#         self.stft_hop = int(hp.data.stft_hop * hp.data.sr)

#     def __len__(self):
#         return self.speaker_num

#     def __getitem__(self, index):

#         selected_utter = []
#         if self.use_random_len and self.cnt % hp.train.N == 0:
#             self.frame_len = random.choice(np.arange(hp.train.average_frame_num - 20, hp.train.average_frame_num + 21, step=5))
#             self.frame_len = int((self.frame_len - 1) * hp.data.stft_hop * hp.data.sr)
#             self.frame_hop = int(self.frame_len * hp.train.frame_hop)
#             self.cnt += 1
#         elif self.use_random_len is False:
#             self.frame_len = hp.train.average_frame_num  # frame
#             self.frame_len = int((self.frame_len - 1) * hp.data.stft_hop * hp.data.sr)  # sample
#             self.frame_hop = int(self.frame_len * hp.train.frame_hop)  # sample

#         utter, _ = np.load(os.path.join(self.data_dir, self.files[index]), allow_pickle=True)
#         # utter = torch.load(os.path.join(self.data_dir, self.files[index]))
#         # utter = utter.squeeze().numpy()
#         max_len = int((utter.shape[0] - self.frame_len) // self.frame_hop + 1)
#         sample_num = int(min(self.set_len, max_len))
#         select_idx = random.sample(range(sample_num), self.M)
#         for idx in select_idx:
#             selected_utter.append(utter[self.frame_hop * idx: self.frame_hop * idx + self.frame_len])
#         selected_utter = torch.from_numpy(np.stack(selected_utter, axis=0)).cuda(0) 
#         selected_utter = self.transform[self.mode](selected_utter)
#         if 'power2db' in self.transform:
#             selected_utter = self.transform['power2db'](selected_utter) 
        
#         return selected_utter.permute(0, 2, 1)


# class my_dataset_test(data.Dataset):
#     def __init__(self, data_dir, transform, enroll_num, verify_num, mode='spectrogram', frame_num=160, enroll_hop=0.5, verify_hop=0.8):
#         super(my_dataset_test, self).__init__()
#         self.data_dir = data_dir
#         self.transform = transform
#         self.enroll_num = enroll_num
#         self.verify_num = verify_num
#         self.mode = mode
#         self.frame_num = frame_num

#         self.files = list(filter(lambda x: fnmatch.fnmatch(x, 's*.npy'), os.listdir(self.data_dir)))  # and x.startswith('p')
#         self.frame_num = int((self.frame_num - 1) * hp.data.stft_hop * hp.data.sr)
#         self.enroll_hop = int(self.frame_num * enroll_hop)  
#         self.verify_hop = int(self.frame_num * verify_hop)

#         self.read_info()

#     def read_info(self):
#         if os.path.exists(os.path.join(self.data_dir, 'info.txt')):
#             with open(os.path.join(self.data_dir, 'info.txt'), 'r') as f:
#                 self.speaker_num = int(f.readline().split(',')[1])
#             f.close()
#         else:
#             self.speaker_num = len(self.files)

#     def __len__(self):
#         return self.speaker_num

#     def __getitem__(self, index):
#         selected_utter = []
#         utter = np.load(os.path.join(self.data_dir, self.files[index]), allow_pickle=True)
#         # utter = torch.load(os.path.join(self.data_dir, self.files[index]))
#         # if isinstance(utter, tuple):
#         #     utter = utter[0].squeeze().numpy()
#         # else:
#         #     utter = utter.squeeze().numpy()

#         starter = random.randint(0, len(utter) - ((self.enroll_num - 1) * self.enroll_hop + self.frame_num + 1))
#         for idx in range(self.enroll_num):
#             selected_utter.append(utter[starter + idx * self.enroll_hop: starter + idx * self.enroll_hop + self.frame_num])
#         ender = starter + idx * self.enroll_hop + self.frame_num
#         utter = np.concatenate((utter[0: starter], utter[ender: ]), axis=0)

#         sample_num = (len(utter) - self.frame_num) // self.verify_hop + 1
#         assert sample_num > self.verify_num
#         utter_index = random.sample(range(0, sample_num), self.verify_num)
#         for idx in utter_index:
#             selected_utter.append(utter[self.verify_hop * idx: self.verify_hop * idx + self.frame_num])

#         selected_utter = torch.from_numpy(np.stack(selected_utter, axis=0))
#         selected_utter = self.transform[self.mode](selected_utter)  # .detach()
#         selected_utter = 10 * torch.log10(torch.clamp(selected_utter, 1e-10))
#         return selected_utter.permute(0, 2, 1)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Created on Mon Aug  6 20:55:52 2018

# @author: harry
# """
# import glob
# import numpy as np
# import os
# import random
# from random import shuffle
# import torch
# from torch.utils.data import Dataset

# from config.hparam import hparam as hp


# class timit_dataset(Dataset):
    
#     def __init__(self, shuffle=True, utter_start=0):
        
#         # data path
#         if hp.training:
#             self.path = hp.data.train_path
#             self.utter_num = hp.train.M
#         else:
#             self.path = hp.data.test_path
#             self.utter_num = hp.test.M
#         self.file_list = os.listdir(self.path)
#         self.shuffle=shuffle
#         self.utter_start = utter_start
        
#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
        
#         np_file_list = os.listdir(self.path)
        
#         if self.shuffle:
#             selected_file = random.sample(np_file_list, 1)[0]  # select random speaker
#         else:
#             selected_file = np_file_list[idx]               
        
#         utters = np.load(os.path.join(self.path, selected_file), allow_pickle=True)        # load utterance spectrogram of selected speaker
#         if self.shuffle:
#             utter_index = np.random.randint(0, utters.shape[0], self.utter_num)   # select M utterances per speaker
#             utterance = utters[utter_index]       
#         else:
#             utterance = utters[self.utter_start: self.utter_start+self.utter_num] # utterances of a speaker [batch(M), n_mels, frames]

#         utterance = utterance[:,:,:160]               # TODO implement variable length batch size

#         utterance = torch.tensor(np.transpose(utterance, axes=(0,2,1)))     # transpose [batch, frames, n_mels]
#         return utterance

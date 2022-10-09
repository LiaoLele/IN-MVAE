import torch.utils.data as data
import soundfile as sf
import pandas as pd
import numpy as np
import librosa
import random
import torch
import copy
import os


class CleanDataset(data.Dataset):
    def __init__(self, csv_paths, usecols, data_prefix):
        super(CleanDataset, self).__init__()
        # 'usecols': ['RelativePath', 'Offset', 'Duration', 'Sr']

        self.data_prefix = data_prefix
        self.arks = []
        self.ark2idx = {}
        negs = 0
        for ark_id, path in enumerate(csv_paths):
            df = pd.read_csv(path, header=0, index_col=None, usecols=usecols)
            self.arks.append(df)
            self.ark2idx[ark_id] = list(range(negs, negs + df.shape[0]))
            negs += df.shape[0]
        self.egs = pd.concat(self.arks, axis=0, ignore_index=True, copy=True)
        self.len = self.egs.shape[0]
        assert self.len == negs == self.ark2idx[len(csv_paths) - 1][-1] + 1

    @property
    def spkr_num(self):
        spkr_set = set(self.egs['SpkrID'])
        return len(spkr_set)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        spkr_id, uttpath, offset, dur, sr = self.egs.loc[index]
        uttpath = os.path.join(self.data_prefix, uttpath)
        s, sr_raw = sf.read(uttpath, start=offset, stop=offset + dur)
        if sr_raw != sr:
            s = librosa.core.resample(s, sr_raw, sr)
        s = s / np.max(np.abs(s))
        utt = torch.from_numpy(s)
        return utt


class NoisyDataset(data.Dataset):

    def __init__(self, arkinfo_path, usecols_ark, usecols_egs, file_prefix, data_prefix, extra_func=None):
        super(NoisyDataset, self).__init__()      
        self.data_prefix = data_prefix
        self.extra_func = extra_func
        ark_df = pd.read_csv(arkinfo_path, header=0, index_col=None, usecols=usecols_ark)
        self.ark_num = ark_df.shape[0]
        self.csv_paths = [os.path.join(file_prefix, path) for path in ark_df[usecols_ark[0]]]
        self.negs_per_ark = list(ark_df[usecols_ark[1]])
        self.nspkr_per_eg = list(ark_df[usecols_ark[2]])
        self.ark_remark = list(ark_df[usecols_ark[3]])
        self.len = sum(self.negs_per_ark)
        self.encode()

        self.data_list = [[] for _ in range(self.ark_num)]
        for ark_id in range(self.ark_num):
            excel_reader = pd.ExcelFile(self.csv_paths[ark_id])
            for idx in range(self.nspkr_per_eg[ark_id]):
                df = excel_reader.parse(sheet_name=str(idx), header=0, usecols=usecols_egs)
                self.data_list[ark_id].append(df)
            if self.ark_remark[ark_id] == 'permute':
                df = excel_reader.parse(sheet_name='SplitInfo', header=0, usecols=['SplitRange'])
                self.data_list[ark_id].append(df)

    @property
    def spkr_num(self):
        spkr_set = set()
        for i in range(self.ark_num):
            if self.nspkr_per_eg[i] == 1:
                spkr_set.update(set(self.data_list[i][0]['SpkrID']))
            elif self.nspkr_per_eg[i] == 2:
                spkr_set.update(set(self.data_list[i][0]['SpkrID']))
                spkr_set.update(set(self.data_list[i][1]['SpkrID']))
        return len(spkr_set)

    def encode(self):
        self.idx2ark = {}
        self.ark2idx = {}
        idx = 0
        for ark_id in range(self.ark_num):
            for local_id in range(self.negs_per_ark[ark_id]):
                self.idx2ark[idx] = (ark_id, local_id)
                self.ark2idx[ark_id] = idx
                idx += 1
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        ark_id, local_id = self.idx2ark[index]
        if self.ark_remark[ark_id] == 'pure':
            spkr_id, uttpath, offset, dur, sr = self.data_list[ark_id][0].loc[local_id]
            spkr_id_target = spkr_id
            uttpath = os.path.join(self.data_prefix, uttpath)
            s, sr_raw = sf.read(uttpath, start=offset, stop=offset + dur)
            if sr_raw != sr:
                s = librosa.core.resample(s, sr_raw, sr)
            s = s / np.max(np.abs(s))
            utt = [torch.from_numpy(s) for _ in range(2)]
            mix_info = None
        elif self.ark_remark[ark_id] == 'permute':
            utt=[]
            for index_this, idx in enumerate(range(self.nspkr_per_eg[ark_id])):
                spkr_id, uttpath, offset, dur, sr = self.data_list[ark_id][idx].loc[local_id]
                if index_this == 0:
                    spkr_id_target = spkr_id
                uttpath = os.path.join(self.data_prefix, uttpath)
                s, sr_raw = sf.read(uttpath, start=offset, stop=offset + dur)
                if sr_raw != sr:
                    s = librosa.core.resample(s, sr_raw, sr)
                s = s / np.max(np.abs(s))
                utt.append(torch.from_numpy(s))
            mix_info = self.data_list[ark_id][-1].loc[local_id][0]

        elif self.ark_remark[ark_id] == 'mix':
            mix_info = torch.from_numpy(self.extra_func['mix']()).unsqueeze(1)
            utt = []
            for index_this, idx in enumerate(range(self.nspkr_per_eg[ark_id])):
                spkr_id, uttpath, offset, dur, sr = self.data_list[ark_id][idx].loc[local_id]
                if index_this == 0:
                    spkr_id_target = spkr_id
                uttpath = os.path.join(self.data_prefix, uttpath)
                s, sr_raw = sf.read(uttpath, start=offset, stop=offset + dur)
                if sr_raw != sr:
                    s = librosa.core.resample(s, sr_raw, sr)
                s = s / np.max(np.abs(s))
                utt.append(torch.from_numpy(s))
        return self.ark_remark[ark_id], utt[0], utt[1], mix_info


class NoisyDatasetNew(data.Dataset):

    def __init__(self, arkinfo_path, usecols_ark, usecols_egs, file_prefix, data_prefix, extra_func=None, nsubutt_per_utt='all'):
        super(NoisyDatasetNew, self).__init__()
        # usecols_ark = ['RelativePath', 'NumberOfEgsPerArk', 'NumberOfSpeakersPerEgs', 'Remark']        

        self.data_prefix = data_prefix
        self.extra_func = extra_func
        ark_df = pd.read_csv(arkinfo_path, header=0, index_col=None, usecols=usecols_ark)
        self.ark_num = ark_df.shape[0]
        self.csv_paths = [os.path.join(file_prefix, path) for path in ark_df[usecols_ark[0]]]
        self.negs_per_ark = list(ark_df[usecols_ark[1]])
        self.nspkr_per_eg = list(ark_df[usecols_ark[2]])
        self.nutt_per_spkr = list(ark_df[usecols_ark[3]])
        self.ark_remark = list(ark_df[usecols_ark[4]])
        self.len = sum(self.negs_per_ark)
        self.encode()

        self.data_list = [[] for _ in range(self.ark_num)]
        self.nutt_per_spkr_list = []
        for ark_id in range(self.ark_num):
            excel_reader = pd.ExcelFile(self.csv_paths[ark_id])
            self.nutt_per_spkr_list.append(list(map(int, self.nutt_per_spkr[ark_id].split(':'))))
            tmp = []
            i = 0
            for num in self.nutt_per_spkr_list[ark_id]:
                tmp.append(list(range(i, i + num)))
                i += num
            if isinstance(nsubutt_per_utt, int):
                self.nutt_per_spkr_list[ark_id] = [t if t <= nsubutt_per_utt else nsubutt_per_utt for t in self.nutt_per_spkr_list[ark_id]]
            for idx in range(self.nspkr_per_eg[ark_id]):
                df_idx_list = random.sample(tmp[idx], k=self.nutt_per_spkr_list[ark_id][idx])
                for df_idx in df_idx_list:
                    df = excel_reader.parse(sheet_name=str(df_idx), header=0, usecols=usecols_egs)
                    self.data_list[ark_id].append(df)
            if self.ark_remark[ark_id] == 'permute':
                df = excel_reader.parse(sheet_name='SplitInfo', header=0, usecols=['SplitRange'])
                self.data_list[ark_id].append(df)

    @property
    def spkr_num(self):
        spkr_set = set()
        for i in range(self.ark_num):
            if self.nspkr_per_eg[i] == 1:
                spkr_set.update(set(self.data_list[i][0]['SpkrID']))
            elif self.nspkr_per_eg[i] == 2:
                spkr_set.update(set(self.data_list[i][0]['SpkrID']))
                spkr_set.update(set(self.data_list[i][1]['SpkrID']))
        return len(spkr_set)

    def encode(self):
        self.idx2ark = {}
        self.ark2idx = {}
        idx = 0
        for ark_id in range(self.ark_num):
            for local_id in range(self.negs_per_ark[ark_id]):
                self.idx2ark[idx] = (ark_id, local_id)
                self.ark2idx[ark_id] = idx
                idx += 1
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        ark_id, local_id = self.idx2ark[index]
        nutt_per_spkr = self.nutt_per_spkr_list[ark_id]
        utt = []
        utt_idx = 0
        if self.ark_remark[ark_id] == 'pure':
            for idx, utt_id in enumerate(range(self.nspkr_per_eg[ark_id])):
                utt_tmp = []
                for j in range(nutt_per_spkr[idx]):
                    spkr_id, uttpath, offset, dur, sr = self.data_list[ark_id][utt_idx].loc[local_id]
                    uttpath = os.path.join(self.data_prefix, uttpath)
                    s, sr_raw = sf.read(uttpath, start=offset, stop=offset + dur)
                    if sr_raw != sr:
                        s = librosa.core.resample(s, sr_raw, sr)
                    s = s / np.max(np.abs(s))
                    utt_tmp.append(torch.from_numpy(s))
                    utt_idx += 1
                utt_tmp = torch.stack(utt_tmp, axis=0)
                if idx == 0:
                    spkr_id_target = spkr_id
                utt.append(utt_tmp)
            assert utt_idx == sum(nutt_per_spkr)
            utt.append(utt[0].clone())
            mix_info = None
        elif self.ark_remark[ark_id] == 'permute':
            for idx, utt_id in enumerate(range(self.nspkr_per_eg[ark_id])):
                utt_tmp = []
                for j in range(nutt_per_spkr[idx]):
                    spkr_id, uttpath, offset, dur, sr = self.data_list[ark_id][utt_idx].loc[local_id]
                    uttpath = os.path.join(self.data_prefix, uttpath)
                    s, sr_raw = sf.read(uttpath, start=offset, stop=offset + dur)
                    if sr_raw != sr:
                        s = librosa.core.resample(s, sr_raw, sr)
                    s = s / np.max(np.abs(s))
                    utt_tmp.append(torch.from_numpy(s))
                    utt_idx += 1
                utt_tmp = torch.stack(utt_tmp, axis=0)
                if idx == 0:
                    spkr_id_target = spkr_id
                utt.append(utt_tmp)
            assert utt_idx == sum(nutt_per_spkr)
            mix_info = self.data_list[ark_id][-1].loc[local_id][0]
        elif self.ark_remark[ark_id] == 'mix':
            mix_info = torch.from_numpy(self.extra_func['mix']()).unsqueeze(1)
            for idx, utt_id in enumerate(range(self.nspkr_per_eg[ark_id])):
                utt_tmp = []
                for j in range(nutt_per_spkr[idx]):
                    spkr_id, uttpath, offset, dur, sr = self.data_list[ark_id][utt_idx].loc[local_id]
                    uttpath = os.path.join(self.data_prefix, uttpath)
                    s, sr_raw = sf.read(uttpath, start=offset, stop=offset + dur)
                    if sr_raw != sr:
                        s = librosa.core.resample(s, sr_raw, sr)
                    s = s / np.max(np.abs(s))
                    utt_tmp.append(torch.from_numpy(s))
                    utt_idx += 1
                utt_tmp = torch.stack(utt_tmp, axis=0)
                if idx == 0:
                    spkr_id_target = spkr_id
                utt.append(utt_tmp)
            assert utt_idx == sum(nutt_per_spkr)
        return self.ark_remark[ark_id], utt, mix_info, torch.tensor(spkr_id_target), nutt_per_spkr


class InferenceDataset(data.Dataset):

    def __init__(self, arkinfo_path, usecols_ark, usecols_egs, file_prefix, data_prefix, extra_func=None):
        super(NoisyDataset, self).__init__()
        # usecols_ark = ['RelativePath', 'NumberOfEgsPerArk', 'NumberOfSpeakersPerEgs']        

        self.data_prefix = data_prefix
        self.extra_func = extra_func
        ark_df = pd.read_csv(arkinfo_path, header=0, index_col=None, usecols=usecols_ark)
        self.ark_num = ark_df.shape[0]
        self.csv_paths = [os.path.join(file_prefix, path) for path in ark_df[usecols_ark[0]]]
        self.negs_per_ark = list(ark_df[usecols_ark[1]])
        self.nspkr_per_eg = list(ark_df[usecols_ark[2]])
        self.ark_remark = list(ark_df[usecols_ark[3]])
        self.len = sum(self.negs_per_ark)
        self.encode()

        self.data_list = [[] for _ in range(self.ark_num)]
        for ark_id in range(self.ark_num):
            excel_reader = pd.ExcelFile(self.csv_paths[ark_id])
            for spkr_id in range(self.nspkr_per_eg[ark_id]):
                df = excel_reader.parse(sheet_name=str(spkr_id), header=0, usecols=usecols_egs)
                self.data_list[ark_id].append(df)
            if self.ark_remark[ark_id] == 'permute':
                df = excel_reader.parse(sheet_name='SplitInfo', header=0, usecols=['SplitRange'])
                self.data_list[ark_id].append(df)

    @property
    def spkr_num(self):
        spkr_set = set()
        for i in range(self.ark_num):
            if self.nspkr_per_eg[i] == 1:
                spkr_set.update(set(self.data_list[i][0]['SpkrID']))
            elif self.nspkr_per_eg[i] == 2:
                spkr_set.update(set(self.data_list[i][0]['SpkrID']))
                spkr_set.update(set(self.data_list[i][1]['SpkrID']))
        return len(spkr_set)

    def encode(self):
        self.idx2ark = {}
        self.ark2idx = {}
        idx = 0
        for ark_id in range(self.ark_num):
            for local_id in range(self.negs_per_ark[ark_id]):
                self.idx2ark[idx] = (ark_id, local_id)
                self.ark2idx[ark_id] = idx
                idx += 1
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        ark_id, local_id = self.idx2ark[index]
        if self.ark_remark[ark_id] == 'pure':
            spkr_id, uttpath, offset, dur, sr = self.data_list[ark_id][0].loc[local_id]
            uttpath = os.path.join(self.data_prefix, uttpath)
            s, sr_raw = sf.read(uttpath, start=offset, stop=offset + dur)
            if sr_raw != sr:
                s = librosa.core.resample(s, sr_raw, sr)
            s = s / np.max(np.abs(s))
            utt = [torch.from_numpy(s) for _ in range(2)]
            mix_info = None
        elif self.ark_remark[ark_id] == 'permute':
            utt = []
            for utt_id in range(self.nspkr_per_eg[ark_id]):
                spkr_id, uttpath, offset, dur, sr = self.data_list[ark_id][utt_id].loc[local_id]
                uttpath = os.path.join(self.data_prefix, uttpath)
                s, sr_raw = sf.read(uttpath, start=offset, stop=offset + dur)
                if sr_raw != sr:
                    s = librosa.core.resample(s, sr_raw, sr)
                s = s / np.max(np.abs(s))
                utt.append(torch.from_numpy(s))
            mix_info = self.data_list[ark_id][-1].loc[local_id][0]
        elif self.ark_remark[ark_id] == 'mix':
            mix_info = torch.from_numpy(self.extra_func['mix']()).unsqueeze(1)
            utt = []
            for utt_id in range(self.nspkr_per_eg[ark_id]):
                spkr_id, uttpath, offset, dur, sr = self.data_list[ark_id][utt_id].loc[local_id]
                uttpath = os.path.join(self.data_prefix, uttpath)
                s, sr_raw = sf.read(uttpath, start=offset, stop=offset + dur)
                if sr_raw != sr:
                    s = librosa.core.resample(s, sr_raw, sr)
                s = s / np.max(np.abs(s))
                utt.append(torch.from_numpy(s))
        return self.ark_remark[ark_id], spkr_id, utt[0], utt[1], mix_info


class TrainBatchSampler(data.Sampler):

    def __init__(self, ark2idx, batch_size, n_batch=None, drop_last=True, shuffle=True):
        """ 
        Args:
            `n_batch`: number of batches
        """
        self.ark2idx = ark2idx
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.n_batch = self.get_len(n_batch)
        self.shuffle = shuffle

    @classmethod
    def from_dataset(cls, dataset, batch_size, n_batch=None, drop_last=True, shuffle=True):
        ark2idx = copy.deepcopy(dataset.ark2idx)
        return cls(ark2idx, batch_size, n_batch=n_batch, drop_last=drop_last, shuffle=shuffle)

    def get_len(self, n_batch):
        if n_batch is None:
            n_batch = 0
            for ark_id, utt_id_list in self.ark2idx.items():
                if self.drop_last:
                    n_batch += int(len(utt_id_list) // self.batch_size)
                else:
                    n_batch = n_batch + int(len(utt_id_list) // self.batch_size) + int(len(utt_id_list) % self.batch_size > 0)
        return n_batch

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        batch_list = []
        ark2idx_copy = copy.deepcopy(self.ark2idx)
        for ark_idx, utt_id_list in ark2idx_copy.items():
            if self.shuffle:
                random.shuffle(utt_id_list)
            for i in range(int(len(utt_id_list) // self.batch_size)):
                batch_list.append(utt_id_list[i * self.batch_size: i * self.batch_size + self.batch_size])
            if (not self.drop_last) and (len(utt_id_list) % self.batch_size > 0):
                batch_list.append(utt_id_list[(i + 1) * self.batch_size:])
        if self.shuffle:
            random.shuffle(batch_list)
        for batch in batch_list[0: self.n_batch]:
            yield batch

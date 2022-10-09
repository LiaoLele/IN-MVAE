import os
import glob
import random
import pickle
import librosa
import datetime
import numpy as np
import pandas as pd
import soundfile as sf
from collections import defaultdict


def AddCols(csv_paths, col_name, col_val, col_insert_idx, pd_ori_cols):
    pd_new_cols = pd_ori_cols.copy()
    pd_new_cols.insert(col_insert_idx, col_name)
    for path in csv_paths:
        df = pd.read_csv(path, header=0, index_col=None, usecols=pd_ori_cols)
        df[col_name] = [col_val for _ in range(df.shape[0])]
        df = df.reindex(columns=pd_new_cols)
        df.to_csv(path, index_label='Index')



def CombineCSVs(csv_paths, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    f_readme = open(os.path.join(out_dir, 'Readme-CombineCSVs.txt'), 'wt')
    df_list = []
    for path in csv_paths:
        df_list.append(pd.read_csv(path, header=0, index_col=None))
        print(f"From {path}", file=f_readme)
    df = pd.concat(df_list, axis=0, ignore_index=True)
    """ Need modification """
    df.to_csv(os.path.join(out_dir, 'metainfo.csv'), columns=list(df.columns)[1:], index_label='Index')

    max_utt_len = max(df['LenInSamples'])
    min_utt_len = min(df['LenInSamples'])
    sr = df.at[0, 'Sr']
    print(f'Number of keys are {df.shape[0]}', file=f_readme)
    print(f"Max utterance len is {round(max_utt_len / sr / 60, 2)}", file=f_readme)
    print(f"Min utterance len is {round(min_utt_len / sr / 60, 2)}", file=f_readme)
    print(f"Meta Infomation is in {os.path.join(out_dir, 'metainfo.csv')}", file=f_readme)
    f_readme.close()


def ExtractSexualInfo(datainfo_path, out_path):
    df = pd.read_table(datainfo_path, sep='|', header=11, usecols=[0, 1, 2])
    # usecols=['ID','SEX','SUBSET']
    for colname in df.columns.values.tolist():
        colname_new = colname.strip()
        df.rename(columns={colname: colname_new}, inplace=True)
    out_df = {}
    subset_set = set([setname.strip() for setname in list(df['SUBSET'])])
    for setname in subset_set:
        spkrid_female = [str(df[';ID'][i]) for i in range(df.shape[0]) if df['SEX'][i].strip() == 'F' and df['SUBSET'][i].strip() == setname]
        spkrid_male = [str(df[';ID'][i]) for i in range(df.shape[0]) if df['SEX'][i].strip() == 'M' and df['SUBSET'][i].strip() == setname]
        out_df[setname] = pd.concat([pd.DataFrame({'Female': spkrid_female}), pd.DataFrame({'Male': spkrid_male})], axis=1)
    writer = pd.ExcelWriter(out_path)
    for key in out_df.keys():
        out_df[key].to_excel(writer, sheet_name=key, index_label='Index')
    writer.save()
    writer.close()


def Concat2Utt(csv_path, out_dir, uttlen_range, global_offset=None, filter_cond=None, usecols=None, seed=None, suffix=''):
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
    os.makedirs(out_dir, exist_ok=True)
    
    f_readme = open(os.path.join(out_dir, '.'.join(['Readme-Concat2Utt', suffix, 'txt'])), 'wt')
    if seed is None:
        seed = datetime.datetime.now()
    random.seed(seed)
    spk2utt = defaultdict(list)
    df = pd.read_csv(csv_path, header=0, index_col=None, usecols=usecols)
    fs = df.at[0, 'Sr']
    ori_keynum = df.shape[0]

    print(f"Original dataset: {csv_path}", file=f_readme)
    # filter dataset according to min_uttlen
    if filter_cond is not None:
        df = df.query(eval(filter_cond))
        print(f"Filter condition of the original dataset: {filter_cond}", file=f_readme)
        print(f"Number of deprecated keys: {ori_keynum - df.shape[0]}", file=f_readme)
    print(f"Number of final keys: {df.shape[0]}", file=f_readme)
    print(f"Utterance range [s]: {uttlen_range}", file=f_readme)
    print(f"Gloabal offset is : {global_offset} s", file=f_readme)
    print(f"Random Seed: {seed}", file=f_readme)

    global_offset = int(global_offset * fs)
    uttlen_range = [int(x * fs) for x in uttlen_range]
    for abs_spkr_id, (_, spkr_path, spkr_dur, spkr_id, _) in enumerate(df.itertuples()):
        spkr_id = int(spkr_id)
        accumulated_len = global_offset
        while True:
            if (spkr_dur - accumulated_len) < uttlen_range[0]:
                print(f"Finish processing the {abs_spkr_id}th speaker!")
                break
            uttlen = random.randint(uttlen_range[0], min(uttlen_range[1], spkr_dur - accumulated_len))
            spk2utt[spkr_id].append((spkr_path, accumulated_len, uttlen))
            accumulated_len += uttlen

    outpath = os.path.join(out_dir, '.'.join(['spk2utt', suffix, 'pkl']))
    print(f"spk2utt saved in {outpath}", file=f_readme)
    with open(outpath, 'wb') as f_spk2utt:
        pickle.dump(spk2utt, f_spk2utt)
    f_readme.close()
    

class RawDataConcat(object):
    def __init__(self, file_structure, out_prefix, out_rel_dir, out_format='.wav', sr=16000):
        self.file_structure = file_structure
        self.out_prefix = out_prefix
        self.out_rel_dir = out_rel_dir
        self.out_format = out_format
        self.sr = sr
        os.makedirs(os.path.join(self.out_prefix, self.out_rel_dir), exist_ok=True)

        self.paths = []
        for file in self.file_structure:
            self.paths.extend(glob.glob(file))
    def GetKey2path(self):
        pass
    def GetConcateData(self, amplitude_norm=True, mean_norm=False, trim=False, topdb=30):
        self.f_readme = open(os.path.join(self.out_prefix, self.out_rel_dir, 'Readme-RawDataConcat.txt'), 'wt')
        self.key2path, naming_rule = self.GetKey2path()
        sorted_keys = sorted(self.key2path.keys())
        print(f"Concate dataset from {self.file_structure}", file=self.f_readme)
        print(f"Perform Mean Normalization: {mean_norm}", file=self.f_readme)
        print(f"Perform Amplitude Normalization: {amplitude_norm}", file=self.f_readme)
        print(f"Trim Silent segments: {trim};  TopdB: {topdb}", file=self.f_readme)
        print(f'Number of keys are {len(self.key2path.keys())}', file=self.f_readme)
        print(f'Number of keys are {len(self.key2path.keys())}')

        csv_info = []
        max_utt_len = float('-inf')
        min_utt_len = float('inf')
        for idx, key in enumerate(sorted_keys):
            utt_list = []
            if trim:
                trim_utt_list = []
            for path in self.key2path[key]:
                signal, sr_raw = librosa.core.load(path, sr=self.sr)
                if sr_raw != self.sr:
                    signal = librosa.core.resample(signal, sr_raw, self.sr)
                if mean_norm:
                    signal = signal - np.mean(signal)
                if amplitude_norm:
                    signal = signal / np.max(np.abs(signal))
                utt_list.append(signal)
            utt_list = np.concatenate(utt_list)
            if trim:
                intervals = librosa.effects.split(utt_list, top_db=topdb)
                for interval in intervals:
                    trim_utt_list.append(utt_list[interval[0]: interval[1]])
                utt_list = np.concatenate(trim_utt_list, axis=0)
            print(f"Finished processing {eval(naming_rule)}!")
            sf.write(os.path.join(self.out_prefix, self.out_rel_dir, eval(naming_rule) + self.out_format), utt_list, self.sr)

            utt_len = utt_list.shape[0]
            max_utt_len = max_utt_len if max_utt_len >= utt_len else utt_len
            min_utt_len = min_utt_len if min_utt_len <= utt_len else utt_len
            csv_info.append([
                os.path.join(self.out_rel_dir, eval(naming_rule) + self.out_format),
                utt_len,
                str(key),
                self.sr,
                round(utt_len / self.sr / 60, 2),
            ])
        pd_cols = ['RelativePath', 'LenInSamples', 'SpkrID', 'Sr', 'LenInMin']
        df = pd.DataFrame(csv_info, columns=pd_cols)
        df.to_csv(os.path.join(self.out_prefix, self.out_rel_dir, 'metainfo.csv'), index_label='Index')
        print(f"Max utterance len is {round(max_utt_len / self.sr / 60, 2)} min", file=self.f_readme)
        print(f"Min utterance len is {round(min_utt_len / self.sr / 60, 2)} min", file=self.f_readme)
        print(f"Meta Infomation is in {os.path.join(self.out_prefix, self.out_rel_dir, 'metainfo.csv')}", file=self.f_readme)
        self.f_readme.close()


class LibrispeechConcat(RawDataConcat):

    def GetKey2path(self):
        key2path = defaultdict(list)
        for path in self.paths:
            speaker_id = int(os.path.basename(path).rsplit('.')[0].split('-')[0])
            key2path[speaker_id].append(path)
        naming_rule = "'speaker-{:04d}'.format(key)"
        return key2path, naming_rule

 
def main(state=0, sub_state=0):
    if state == 0:
        file_structure = [
            '/home/user/zhaoyi.gu/mnt/g4/librispeech/LibriSpeech/test-clean/*/*/*.flac',
            # '/home/user/zhaoyi.gu/mnt/g4/librispeech/LibriSpeech/train-clean-360/*/*/*.flac',
        ]
        out_prefix = '/data/ssd1/zhaoyi.gu'
        out_rel_dir = 'Librispeech/test/ampnorm_meannorm_trim30'
        out_format = '.wav'
        sr = 16000
        Libri = LibrispeechConcat(file_structure, out_prefix, out_rel_dir, out_format=out_format, sr=sr)

        if sub_state == 0:
            amplitude_norm = True
            mean_norm = True
            trim = True
            topdb = 30
            Libri.GetConcateData(amplitude_norm=amplitude_norm, mean_norm=mean_norm, trim=trim, topdb=topdb)
    
    if state == 1:
        csv_paths = [
            '/data/ssd1/zhaoyi.gu/Librispeech/test/ampnorm_meannorm_trim30/metainfo.csv',
            '/data/ssd1/zhaoyi.gu/Librispeech/dev/ampnorm_meannorm_trim30/metainfo.csv',
        ]
        out_dir = '/data/ssd1/zhaoyi.gu/Librispeech/test_dev/ampnorm_meannorm_trim30'
        CombineCSVs(csv_paths, out_dir)

    if state == 2:
        csv_paths = [
            '/data/ssd1/zhaoyi.gu/Librispeech/train/ampnorm_meannorm_trim30/metainfo.csv',
            '/data/ssd1/zhaoyi.gu/Librispeech/test/ampnorm_meannorm/metainfo.csv',
            '/data/ssd1/zhaoyi.gu/Librispeech/dev/ampnorm_meannorm/metainfo.csv',
        ]
        pd_ori_cols = ['RelativePath', 'LenInSamples', 'SpkrID', 'LenInMin']
        col_name = 'Sr'
        col_val = 16000
        col_insert_idx = 3
        AddCols(csv_paths, col_name, col_val, col_insert_idx, pd_ori_cols)

    if state == 3:
        datainfo_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/SPEAKERS.TXT'
        out_path = '/data/hdd0/leleliao/DATASET/LibriSpeech/SexualInfo.xls'
        ExtractSexualInfo(datainfo_path, out_path)

    if state == 4:
        # csv_path = '/data/ssd1/zhaoyi.gu/Librispeech/train/ampnorm_meannorm_trim30/metainfo.csv'
        # out_dir = '/data/hdd0/leleliao/DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/'
        # csv_path = '/data/ssd1/zhaoyi.gu/Librispeech/dev/ampnorm_meannorm_trim30/metainfo.csv'
        # out_dir = '/data/hdd0/leleliao/DATASET/LibriSpeech/dev/ampnorm_meannorm_trim30/'
        # csv_path = '/data/ssd1/zhaoyi.gu/Librispeech/test/ampnorm_meannorm_trim30/metainfo.csv'
        # out_dir = '/data/hdd0/leleliao/DATASET/LibriSpeech/test/ampnorm_meannorm_trim30/'
        csv_path = '/data/ssd1/zhaoyi.gu/Librispeech/test_dev/ampnorm_meannorm_trim30/metainfo.csv'
        out_dir = '/data/hdd0/leleliao/DATASET/LibriSpeech/test_dev/ampnorm_meannorm_trim30/'
        uttlen_range = [5, 30]
        global_offset = 0
        # filter_cond = "'LenInSamples > {}'.format(int(fs * 60 * 3))"
        filter_cond = None
        usecols = ['RelativePath', 'LenInSamples', 'SpkrID', 'Sr']
        seed = datetime.datetime.now()
        suffix = 'clean'

        Concat2Utt(csv_path, out_dir, uttlen_range, global_offset=global_offset, filter_cond=filter_cond, usecols=usecols, seed=seed, suffix=suffix)


if __name__ == "__main__":
    main(state=4, sub_state=0)
    
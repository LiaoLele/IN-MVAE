#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Modified from https://github.com/JanhHyun/Speaker_Verification
import glob
import os
import librosa
import numpy as np
import soundfile as sf
import shutil
import pickle                                


class RawDataConcatenate(object):

    def __init__(self, raw_data_ptn, out_data_dir, sr=16000):
        self.raw_data_dir = glob.glob(os.path.dirname(os.path.dirname(raw_data_ptn))) #返回所有匹配的文件路径列表（list）
        self.raw_data_dir = sorted(self.raw_data_dir)[0:2]
        self.file_suffix = raw_data_ptn.rsplit('.')[1]   # figure out the audio file suffix i.e. "wav", "flac", etc.

        self.out_data_dir = out_data_dir
        os.makedirs(self.out_data_dir, exist_ok=True)

        self.sr = sr

    @staticmethod
    def get_spkr_name(folder_name):
        spkr_id = os.path.basename(folder_name)[3:]
        return spkr_id

    def load_audio(self, path):
        utter, _ = librosa.core.load(path, sr=self.sr)    # load
        utter = utter / np.max(np.abs(utter))  # normalization
        return utter

    def name_ptn(self, spkr_id):
        return eval("\"speaker{:04d}.wav\".format(int(spkr_id))")

    def concatenate(self, use_trim=False, top_db=None):
        """ Concatenate all audio data from the same speaker into one big audio file 
            use_trim: default "False"
            kwargs: parameters used in librosa.effects.split
                    top_db = 30
        """
        spkr_num = len(self.raw_data_dir)
        print("No.speakers : %d" % spkr_num)
        f_info = open(os.path.join(self.out_data_dir, 'info.pkl'), 'wb')  # f to write down information of processed data
        info = []
        f_setting = open(os.path.join(self.out_data_dir, 'setting.txt'), 'wt')   # f to write down settings of processing pipe

        # Concatenating......
        print("Concatenation begins...")
        if use_trim:
            print("Removing nonspeech samples is activated.")
        for i, folder in enumerate(self.raw_data_dir):
            spkr_id = self.get_spkr_name(folder_name=folder)
            print("%d: %sth speaker processing..." % (i, spkr_id))
            utter_all = []
            for root, _, utter_names in os.walk(folder):
                if len(utter_names) > 0:
                    utter_names = list(filter(lambda x: x.endswith(self.file_suffix), utter_names))
                    for utter_name in utter_names:
                        utter_path = os.path.join(root, utter_name)         # absolute path of each utterance
                        utter = self.load_audio(utter_path)
                        utter_all.append(utter)
            utter_all = np.concatenate(utter_all, axis=0)

            if use_trim:
                utter_trim = []
                intervals = librosa.effects.split(utter_all, top_db=top_db)
                for interval in intervals:
                    utter_trim.append(utter_all[interval[0]: interval[1]])
                utter_trim = np.concatenate(utter_trim, axis=0)
                sf.write(os.path.join(self.out_data_dir, self.name_ptn(spkr_id=spkr_id)), utter_trim, self.sr)
                info.append((os.path.join(self.out_data_dir, self.name_ptn(spkr_id=spkr_id)), utter_trim.shape[0], i))
            else:
                sf.write(os.path.join(self.out_data_dir, self.name_ptn(spkr_id=spkr_id)), utter_all, self.sr)
                info.append((os.path.join(self.out_data_dir, self.name_ptn(spkr_id=spkr_id)), utter_all.shape[0], i))
        # Concatenating succeed

        print("Writing necessary files...")
        pickle.dump(info, f_info)
        f_info.close()

        info = sorted(info, key=lambda x: x[1])
        print("Speaker number: {}".format(spkr_num), file=f_setting)
        print("Minimum length: {}\t{}".format(info[0][0], info[0][1]), file=f_setting)
        print("Maximum length: {}\t{}".format(info[-1][0], info[-1][1]), file=f_setting)
        print("sr: {}".format(self.sr), file=f_setting)
        print("Use trim: {}".format(use_trim), file=f_setting)
        print("top_db: {}".format(top_db), file=f_setting)
        f_setting.close()

    def pkl2txt(self):
        f = open(os.path.join(self.out_data_dir, 'DataIdxSprd.pkl'), 'rb')
        info = pickle.load(f)
        f.close()
        fw = open(os.path.join(self.out_data_dir, 'txt_DataIdxSprd.txt'), 'wt')

        for i in range(len(info)):
            for item in info[i]:
                fw.write("{}\t".format(item))
            fw.write("\n")
        fw.close()

    def add2pkl(self):
        '''给wav文件加上绝对路径'''
        f = open(os.path.join(self.out_data_dir, 'info.pkl'), 'rb')
        info = pickle.load(f)
        f.close()

        shutil.copy(os.path.join(self.out_data_dir, 'info.pkl'), os.path.join(self.out_data_dir, 'info_bak.pkl'))
        if os.path.exists(os.path.join(self.out_data_dir, 'info_bak.pkl')):
            print("Copy to .bak succeeded!")

        # Contents to be added!!!
        for i in range(len(info)):
            spec_info = list(info[i])
            spec_info[0] = self.out_data_dir + spec_info[0]
            info[i] = tuple(spec_info)

        f = open(os.path.join(self.out_data_dir, 'info.pkl'), 'wb')
        pickle.dump(info, f)
        f.close()
        # [('/data/hdd0/zhaoyigu/DATASET/Librispeech/concatenate/train_clean/speaker0103.wav', 20059472, 0),
        #  ('/data/hdd0/zhaoyigu/DATASET/Librispeech/concatenate/train_clean/speaker1034.wav', 12053265, 1),
        #  ('/data/hdd0/zhaoyigu/DATASET/Librispeech/concatenate/train_clean/speaker1040.wav', 9290752, 2),...]

if __name__ == "__main__":
    raw_data_dir = '/data/hdd0/zhaoyigu/DATASET/VoxCeleb/raw/dev_wav/*/*/*.wav'
    out_data_dir = '/data/hdd0/zhaoyigu/DATASET/Librispeech/concatenate/train_clean/'
    process = RawDataConcatenate(raw_data_dir, out_data_dir)
    # process.pkl2txt()
    process.add2pkl()

    


""" BACKUPS """
# def write_info(out_dir, sample_len, sample_hop, sp_num):
#     with open(os.path.join(out_dir, 'info.txt'), 'w') as f:
#         f.write('speaker_number,{}\n'.format(str(sp_num)))
#         f.write('sample_len,{}\n'.format(str(sample_len)))
#         f.write('sample_hop,{}\n'.format(str(sample_hop)))
#     f.close()


# def save_big_file():
#     speaker_per_file = 200

#     os.makedirs(hp.data.train_spec_path, exist_ok=True)

#     files = list(filter(lambda x: x.endswith('.npy'), os.listdir(hp.data.train_path)))
#     frame_len = hp.train.average_frame_num + 20  # frame
#     frame_len = int((frame_len - 1) * hp.data.stft_hop * hp.data.sr)  # sample
#     frame_hop = int(frame_len * hp.train.frame_hop)  # sample
#     set_len = hp.data.num_utter_per_speaker
#     utter_set_len = int((set_len - 1) * frame_hop + frame_len)

#     with open(os.path.join(hp.data.train_spec_path, 'info.txt'), 'w') as f:
#         f.write('speaker_per_file,{}\n'.format(speaker_per_file))
#     f.close()

#     big_file, cnt = [], 0
#     for i, file in enumerate(files):
#         print('processing {}'.format(file))
#         utter = np.load(os.path.join(hp.data.train_path, file), allow_pickle=True)
#         assert utter.shape[0] >= utter_set_len
#         big_file.append(utter[0: utter_set_len])
#         if (i + 1) % speaker_per_file == 0:
#             print('writing the {}th file'.format(cnt))
#             big_file = np.stack(big_file, 0)
#             np.save(os.path.join(hp.data.train_spec_path, str(cnt) + '.npy'), big_file)
#             big_file = []
#             cnt += 1
#     if len(big_file) > 0:
#         big_file = np.stack(big_file, 0)
#         np.save(os.path.join(hp.data.train_spec_path, str(cnt) + '.npy'), big_file)        



#         max_len = int((utter.shape[0] - frame_len) // frame_hop + 1)
#         assert max_len >= set_len
#         sample_num = int(min(set_len, max_len))
#         for idx in range(sample_num):
#             selected_utter.append(utter[frame_hop * idx: frame_hop * idx + frame_len])
#         selected_utter = torch.from_numpy(np.stack(selected_utter, axis=0))  # [sample_num, frame_len] [sample]
#         selected_utter = transform[hp.train.feature](selected_utter)  #.detach()
#         if 'power2db' in transform:
#             selected_utter = transform['power2db'](selected_utter)  #(selected_utter + 1e-6).log()
#         big_file.append(selected_utter)

# def concatenate(self, ):
#         """ Concatenate all audio data from the same speaker into one big audio file [.wav]
#         """
#         os.makedirs(hp.data.train_path, exist_ok=True)   # make folder to save train file
#         # os.makedirs(hp.data.test_path, exist_ok=True)    # make folder to save test file

#         # sample_len = int((hp.data.tisv_frame - 1) * hp.data.stft_hop * hp.data.sr)
#         # sample_hop = int(hp.data.tisv_hop * sample_len)

#         total_speaker_num = len(audio_path)
#         # total_train_len = 2 * 60 * hp.data.sr
#         # total_test_len = 15 * 60 * hp.data.sr
#         # test_cnt, test_num = 0, 100

#         # train_speaker_num = (total_speaker_num // 10) * 9 - 1         # split total data 90% train and 10% test
#         # write_info(hp.data.train_path, sample_len, sample_hop, train_speaker_num)
#         # write_info(hp.data.test_path, sample_len, sample_hop, total_speaker_num - train_speaker_num)
#         print("total speaker number : %d" % total_speaker_num)
#         # print("train : %d, test : %d" % (train_speaker_num, total_speaker_num - train_speaker_num))
#         if os.path.exists(os.path.join(hp.data.train_path, 'info.pkl')):
#             f = open(os.path.join(hp.data.train_path, 'info.pkl'), 'rb')
#             info = pickle.load(f)
#         else:
#             info = []
#         for i, folder in enumerate(audio_path):
#             speaker_id = os.path.basename(folder)
#             print("%d: %sth speaker processing..." % (i, speaker_id[3:]))
#             utter_all, utter_trim = [], []
#             for root, _, utter_names in os.walk(folder):
#                 if len(utter_names) > 0:
#                     utter_names = list(filter(lambda x: x.endswith('.wav'), utter_names))
#                     for utter_name in utter_names:
#                         utter_path = os.path.join(root, utter_name)         # path of each utterance
#                         utter, sr = librosa.core.load(utter_path, sr=hp.data.sr)        # load utterance audio
#                         utter = utter / np.max(np.abs(utter))  # normalization
#                         utter_all.append(utter)
#             utter_all = np.concatenate(utter_all, axis=0)
#             # indices = utter_all.shape[0] // total_train_len
#             # if indices == 0:
#             #     utter_train = utter_all
#             #     # utter_test = []
#             # else:
#             #     idx = random.randint(0, indices - 1)
#             #     utter_train = utter_all[idx * total_train_len: idx * total_train_len + total_train_len]
#             #     # utter_test = np.concatenate((utter_all[0: idx * total_train_len], utter_all[idx * total_train_len + total_train_len: ]), axis=0)

#             intervals = librosa.effects.split(utter_all, top_db=30)
#             for interval in intervals:
#                 utter_trim.append(utter_all[interval[0]: interval[1]])
#             utter_trim = np.concatenate(utter_trim, axis=0)
#             sf.write(os.path.join(hp.data.train_path, "speaker{:04d}.wav".format(int(speaker_id[3:]))), utter_trim, hp.data.sr)

#             info.append(('speaker{:04d}.wav'.format(int(speaker_id[3:])), utter_trim.shape[0]))
#         f = open(os.path.join(hp.data.train_path, 'info.pkl'), 'wb')
#         pickle.dump(info, f)
#             # torch.save(utter_trim, os.path.join(hp.data.train_path, "speaker-{:04d}.pt".format(int(speaker_id))))

#             # if len(utter_test) > total_test_len:
#             #     np.save(os.path.join(hp.data.test_path, "speaker-{:04d}.npy".format(int(speaker_id))), utter_test)
#             #     test_cnt += 1

#             # if i < train_speaker_num:      # save spectrogram as numpy file
#             #     np.save(os.path.join(hp.data.train_path, "speaker{:03d}.npy".format(i)), utter_trim)
#             #     # if i == 0:
#             #     #     sf.write(os.path.join(hp.data.train_path, 'speaker001.wav'), utter_trim, 16000)
#             # else:
#             #     np.save(os.path.join(hp.data.test_path, "speaker{:03d}.npy".format(i - train_speaker_num)), utter_trim)
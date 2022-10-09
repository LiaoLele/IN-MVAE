import os
import torch
import pickle
import librosa
import mir_eval
import torchaudio
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.io as sio
from pystoi import stoi
from pypesq import pesq
import statistics as stat
from collections import defaultdict
from scipy import convolve
from itertools import product
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pyroomacoustics.bss import ilrma as ilrma_pra

from dataset import NoisyDataset
# from network_inference import AutoEncoder_clsy as AutoEncoder
# from network_inference import AutoEncoder
from network_cnn import AutoEncoder
# from network_cnn_onehot_inference import cvae_standard as AutoEncoder

from BSSAlgorithm.ilrma import myilrma, ilrma, ilrma_mypra
from PreProcess.utils import PlotSpectrogram
from PreProcess.data_utils import Spectrogram
from PreProcess.data_utils import melspectrogram2wav
from PLDA.joint_probability import probability
import time

class SimpleCollate():
    def __init__(self, batch):
        # data = batch
        # self.data = torch.stack(data, axis=0)

        remark, data_target, data_noise, mixinfo = list(zip(*batch))
        self.remark = remark
        self.data_target = torch.stack(data_target, axis=0)
        self.data_noise = torch.stack(data_noise, axis=0)
        self.mixinfo = mixinfo

    def pin_memory(self):
        # self.data = self.data.pin_memory()

        self.data_target = self.data_target.pin_memory()
        self.data_noise = self.data_noise.pin_memory()
        return self


def CollateFnWrapper(batch):
    return SimpleCollate(batch)


class Transform(object):
    def __init__(self, sigproc_param):
        super(Transform, self).__init__()
        self.trans = Spectrogram(**sigproc_param)
    
    def __call__(self, remark, data_target, data_noise, mix_info):
        out = []
        _, target_spec, _ = self.trans(data_target)
        _, noise_spec, _ = self.trans(data_noise)
        for idx, this_mix_info in enumerate(mix_info):
            rmk = remark[idx]
            if rmk == 'pure':
                out.append(target_spec[idx, :, :].clone())
            elif rmk == 'permute':
                noisy_spec = target_spec[idx, :, :].clone()
                this_mix_info = eval(this_mix_info)
                for block in this_mix_info:
                    noisy_spec[block, :] = noise_spec[idx, block, :]
                out.append(noisy_spec)
            elif rmk == 'mix':
                this_mix_info = this_mix_info.to(data_target.device)
                # print(torch.mean(this_mix_info))
                noisy_spec = (1 - this_mix_info) * target_spec[idx, :, :] + this_mix_info * noise_spec[idx, :, :]
                out.append(noisy_spec)
        out = torch.stack(out, dim=0)
        return target_spec.float(), out.float()

class Inference(object):
    def __init__(self, config):
        self.config = config
        if self.config.info.load_model:
            self.build_model()
            self.load_model()
        self.parse_data()
        self.transform = Spectrogram(**self.config.sigproc)  
        # Must be processed after self.parse_data(), because the fs is updated in self.parse_data()

        self.hop_len = int(self.fs * self.config.sigproc.hop_len)
        self.stft_len = int(self.fs * self.config.sigproc.stft_len)
        self.griffinlim = torchaudio.transforms.GriffinLim(n_fft=self.stft_len, n_iter=32, hop_length=self.hop_len).to(self.config.general.device)
        # Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation.
    def load_model(self):
        state_dict = torch.load(f'{os.path.join(self.config.general.prefix, self.config.path.model)}', map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.to(self.config.general.device)
        return

    def build_model(self):
        self.model = AutoEncoder(self.config)
        self.model.eval()
        # self.model.decoder.use_spkr_affine = self.config.info.use_spkr_affine
        return

    def parse_data(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class NoisyReconstruct(Inference):
    def __init__(self, config, suffix=''):
        super(NoisyReconstruct, self).__init__(config)
        self.noise_type = config.info.noise_type
        self.extra_func = config.dataset.CreateMixWeight
        self.transform_new = Transform(self.config.sigproc)
        self.config.path.abs_out_dir = os.path.join(self.config.path.common_prefix, self.config.path.out_dir, suffix)
        os.makedirs(self.config.path.abs_out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.path.abs_out_dir, 'spectrogram'), exist_ok=True)
        os.system(f"cp {os.path.join(os.path.dirname(__file__), 'config.py')} {os.path.join(self.config.path.abs_out_dir, 'Config.py')}")

    def parse_data(self):
        excel_reader = pd.ExcelFile(os.path.join(self.config.path.common_prefix, self.config.path.data))
        sheet_name_list = excel_reader.sheet_names
        self.src_num = len(sheet_name_list) - 1
        self.df = []
        for i in range(self.src_num):
            self.df.append(excel_reader.parse(sheet_name=sheet_name_list[i], header=0, usecols=self.config.dataset.usecols))
        self.fs = self.df[0].at[0, 'Sr']
        self.config.sigproc.sr = self.fs
        if self.config.info.noise_type == 'permute':
            self.mix_info = excel_reader.parse(sheet_name='SplitInfo', header=0, usecols=['SplitRange'])

    def noisyrec_inference_one_utterance(self, target, source, target_seglen=200, target_seghop=0.5):
        with torch.no_grad():
            spec_amp = self.model.inference(target, source, target_seglen=target_seglen, target_seghop=target_seghop)
        return spec_amp[:, :, :target.shape[-1]]

    def noisyrec_inference(self, num_tests=None, wav_length=None, target_seglen=200, target_seghop=0.5, mode='oracle', enroll_len=None):
        if num_tests is None:
            num_tests = self.df[0].shape[0]
        if wav_length is not None:
            wav_length = int(self.fs * wav_length)
            self.griffinlim.length = wav_length
        if self.df[0].shape[0] < num_tests:
            raise ValueError(f"Number of tests {num_tests} exceeds the number of test data {self.df[0].shape[0]}!")
        self.enroll_len = enroll_len

        for test_id in range(num_tests):
            target_path, target_offset, target_dur, fs = self.df[0].loc[test_id]
            source_path, source_offset, source_dur, fs = self.df[1].loc[test_id]
            if self.noise_type == 'permute':
                mix_info = [self.mix_info.loc[test_id][0]]
            elif self.noise_type == 'mix':
                weight = self.extra_func()
                plt.plot(np.linspace(0, 513, num=513, endpoint=False), weight)
                plt.show()
                mix_info = [torch.from_numpy(weight).float().to(self.config.general.device).unsqueeze(1)]
                print(torch.mean(mix_info[0][0:128]))
                print(torch.mean(mix_info[0][128:]))

            # naming
            target_name = os.path.basename(target_path).split('-')[1].rsplit('.', 1)[0]
            source_name = os.path.basename(source_path).split('-')[1].rsplit('.', 1)[0]
            out_basename = f'target-{target_name}_noise-{source_name}'
            target_sig, raw_sr = sf.read(os.path.join(self.config.path.data_prefix, target_path), start=target_offset, stop=target_offset + target_dur)
            if raw_sr != self.fs:
                target_sig = librosa.core.resample(target_sig, raw_sr, self.fs)
                # raw_sr是ori_sr,self.fs是target_srx
            source_sig, raw_sr = sf.read(os.path.join(self.config.path.data_prefix, source_path), start=source_offset, stop=source_offset + source_dur)
            if raw_sr != self.fs:
                source_sig = librosa.core.resample(source_sig, raw_sr, self.fs)

            target_sig = torch.from_numpy(target_sig / np.max(np.abs(target_sig))).to(self.config.general.device).float().unsqueeze(0)
            source_sig = torch.from_numpy(source_sig / np.max(np.abs(source_sig))).to(self.config.general.device).float().unsqueeze(0)

            target_sig_feat, noisy_sig_feat = self.transform_new([self.noise_type], target_sig, source_sig, mix_info)
            _, source_sig_feat, _ = self.transform(source_sig)
            targetrec_amp = self.noisyrec_inference_one_utterance(target_sig_feat, target_sig_feat, target_seglen=target_seglen, target_seghop=target_seghop)
            noisyrec_amp = self.noisyrec_inference_one_utterance(target_sig_feat, noisy_sig_feat, target_seglen=target_seglen, target_seghop=target_seghop)
            fig_target = PlotSpectrogram(target_sig_feat**0.5, self.fs, self.hop_len)
            fig_source = PlotSpectrogram(source_sig_feat**0.5, self.fs, self.hop_len)
            fig_noisy = PlotSpectrogram(noisy_sig_feat**0.5, self.fs, self.hop_len)
            fig_noisyrec = PlotSpectrogram(noisyrec_amp, self.fs, self.hop_len)
            fig_targetrec = PlotSpectrogram(targetrec_amp, self.fs, self.hop_len)
            # import ipdb; ipdb.set_trace()
            fig_noisy.savefig(os.path.join(self.config.path.abs_out_dir, 'spectrogram', f"Noisy-{target_name}.png"), dpi=300)
            fig_noisyrec.savefig(os.path.join(self.config.path.abs_out_dir, 'spectrogram', f"RecNoisy-{source_name}.png"), dpi=300)
            fig_target.savefig(os.path.join(self.config.path.abs_out_dir, 'spectrogram', f"Target-{target_name}.png"), dpi=300)
            fig_targetrec.savefig(os.path.join(self.config.path.abs_out_dir, 'spectrogram', f"RecTarget-{source_name}.png"), dpi=300)
            fig_source.savefig(os.path.join(self.config.path.abs_out_dir, 'spectrogram', f"Source-{target_name}.png"), dpi=300)

            if mode == 'enroll':
                if enroll_len is not None:
                    enroll_len = int(self.fs * self.enroll_len)
                else:
                    enroll_len = target_dur
                enroll_sig, raw_sr = sf.read(os.path.join(self.config.path.data_prefix, target_path), start=0, stop=enroll_len)
                if raw_sr != self.fs:
                    enroll_sig = librosa.core.resample(enroll_sig, raw_sr, self.fs)
                enroll_sig = torch.from_numpy(enroll_sig / np.max(np.abs(enroll_sig))).to(self.config.general.device).float().unsqueeze(0)
                _, enroll_sig_feat, _ = self.transform(enroll_sig)
                noisyrec_amp_enroll = self.noisyrec_inference_one_utterance(enroll_sig_feat, noisy_sig_feat, target_seglen=target_seglen, target_seghop=target_seghop)
                targetrec_amp_enroll = self.noisyrec_inference_one_utterance(enroll_sig_feat, target_sig_feat, target_seglen=target_seglen, target_seghop=target_seghop)
                fig_noisyrec_enroll = PlotSpectrogram(noisyrec_amp_enroll, self.fs, self.hop_len)
                fig_targetrec_enroll = PlotSpectrogram(targetrec_amp_enroll, self.fs, self.hop_len)
                fig_noisyrec_enroll.savefig(os.path.join(self.config.path.abs_out_dir, 'spectrogram', f"Enroll-noisyrec-{out_basename}.png"), dpi=300)
                fig_targetrec_enroll.savefig(os.path.join(self.config.path.abs_out_dir, 'spectrogram', f"Enroll-targetrec-{out_basename}.png"), dpi=300)

    def __call__(self, config, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.noisyrec_inference(**config)


class VoiceConversion(Inference):
    def __init__(self, config, suffix=''):
        super(VoiceConversion, self).__init__(config)
        self.config.path.abs_out_dir = os.path.join(self.config.path.common_prefix, self.config.path.out_dir, suffix)
        os.makedirs(self.config.path.abs_out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.path.abs_out_dir, 'spectrogram'), exist_ok=True)
        os.makedirs(os.path.join(self.config.path.abs_out_dir, 'audio'), exist_ok=True)
        os.system(f"cp {os.path.join(os.path.dirname(__file__), 'config.py')} {os.path.join(self.config.path.abs_out_dir, 'Config.py')}")

    def parse_data(self):
        excel_reader = pd.ExcelFile(os.path.join(self.config.path.common_prefix, self.config.path.data))
        sheet_name_list = excel_reader.sheet_names
        try:
            sheet_name_list.remove('RIRidx')
        except ValueError:
            pass
        try:
            sheet_name_list.remove('SIR')
        except ValueError:
            pass
        self.src_num = len(sheet_name_list)
        self.df = []
        for i in range(self.src_num):
            self.df.append(excel_reader.parse(sheet_name=sheet_name_list[i], header=0, usecols=self.config.dataset.usecols))
        self.df = self.df[:2]
        self.fs = self.df[0].at[0, 'Sr']
        self.config.sigproc.sr = self.fs

    def vc_inference_one_utterance(self, target, source, src_len, target_seglen=200, target_seghop=0.5):
        with torch.no_grad():
            spec_amp = self.model.inference(target, source, target_seglen=target_seglen, target_seghop=target_seghop)
            spec_amp = spec_amp.squeeze(0)
            wav_data = melspectrogram2wav(spec_amp, src_len)
        return wav_data, spec_amp

    def vc_inference(self, num_tests=None, wav_length=None, target_seglen=200, target_seghop=0.5):
        if num_tests is None:
            num_tests = self.df[0].shape[0]
        if wav_length is not None:
            wav_length = int(self.fs * wav_length)
            self.griffinlim.length = wav_length
        if self.df[0].shape[0] < num_tests:
            raise ValueError(f"Number of tests {num_tests} exceeds the number of test data {self.df[0].shape[0]}!")

        for test_id in range(num_tests):
            target_path, target_offset, target_dur, fs = self.df[0].loc[test_id]
            source_path, source_offset, source_dur, fs = self.df[1].loc[test_id]
            # naming
            target_name = os.path.basename(target_path).split('-')[1].rsplit('.', 1)[0]
            source_name = os.path.basename(source_path).split('-')[1].rsplit('.', 1)[0]
            out_basename = f'voice-{target_name}_content-{source_name}'
            target_sig, raw_sr = sf.read(os.path.join(self.config.path.data_prefix, target_path), start=target_offset, stop=target_offset + target_dur)
            if raw_sr != self.fs:
                target_sig = librosa.core.resample(target_sig, raw_sr, self.fs)
            source_sig, raw_sr = sf.read(os.path.join(self.config.path.data_prefix, source_path), start=source_offset, stop=source_offset + source_dur)
            if raw_sr != self.fs:
                source_sig = librosa.core.resample(source_sig, raw_sr, self.fs)

            target_sig = torch.from_numpy(target_sig / np.max(np.abs(target_sig))).to(self.config.general.device).float().unsqueeze(0)
            source_sig = torch.from_numpy(source_sig / np.max(np.abs(source_sig))).to(self.config.general.device).float().unsqueeze(0)
            src_len = int(source_sig.shape[1])

            _, target_sig_feat, _ = self.transform(target_sig)
            _, source_sig_feat, _ = self.transform(source_sig)
            wav, spec_amp = self.vc_inference_one_utterance(target_sig_feat, source_sig_feat, src_len,
                                                            target_seglen=target_seglen, target_seghop=target_seghop)
            fig_target = PlotSpectrogram(target_sig_feat, self.fs, self.hop_len)
            fig_source = PlotSpectrogram(source_sig_feat, self.fs, self.hop_len)
            fig_out = PlotSpectrogram(spec_amp, self.fs, self.hop_len)
            sf.write(os.path.join(self.config.path.abs_out_dir, 'audio', out_basename + '.wav'), wav.squeeze(), self.fs)
            sf.write(os.path.join(self.config.path.abs_out_dir, 'audio', f'target-{target_name}.wav'), target_sig.detach().cpu().numpy().squeeze(), self.fs)
            sf.write(os.path.join(self.config.path.abs_out_dir, 'audio', f'source-{source_name}.wav'), source_sig.detach().cpu().numpy().squeeze(), self.fs)
            fig_target.savefig(os.path.join(self.config.path.abs_out_dir, 'spectrogram', f"target-{target_name}.png"), dpi=300)
            fig_source.savefig(os.path.join(self.config.path.abs_out_dir, 'spectrogram', f"source-{source_name}.png"), dpi=300)
            fig_out.savefig(os.path.join(self.config.path.abs_out_dir, 'spectrogram', f"{out_basename}.png"), dpi=300)

    def __call__(self, config, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.vc_inference(**config)


class ReconstructSpectrogram(Inference):
    def __init__(self, config, suffix=''):
        super(ReconstructSpectrogram, self).__init__(config)
        self.config = config
        self.config.path.abs_out_dir = os.path.join(self.config.path.common_prefix, self.config.path.out_dir, suffix)
        os.makedirs(self.config.path.abs_out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.path.abs_out_dir, 'spectrogram'), exist_ok=True)
        os.system(f"cp {os.path.join(os.path.dirname(__file__), 'config.py')} {os.path.join(self.config.path.abs_out_dir, 'Config.py')}")

    def parse_data(self):
        excel_reader = pd.ExcelFile(os.path.join(self.config.path.common_prefix, self.config.path.data))
        sheet_name_list = excel_reader.sheet_names
        try:
            sheet_name_list.remove('RIRidx')
        except ValueError:
            pass
        try:
            sheet_name_list.remove('SIR')
        except ValueError:
            pass
        self.src_num = len(sheet_name_list)
        self.df = []
        for i in range(self.src_num):
            self.df.append(excel_reader.parse(sheet_name=sheet_name_list[i], header=0, usecols=self.config.dataset.usecols))
        self.fs = self.df[0].at[0, 'Sr']
        self.config.sigproc.sr = self.fs
        if self.config.dataset.target_utter is not None:
            self.df = self.df[self.config.dataset.target_utter[0]]
        else:
            self.df = self.df[0]

    def rec_inference_one_utterance(self, enroll, target, target_seglen=200, target_seghop=0.5):
        with torch.no_grad():
            spec_amp = self.model.inference(enroll, target, target_seglen=target_seglen, target_seghop=target_seghop)
        return spec_amp[:, :, :target.shape[-1]]

    def rec_inference(self, mode='oracle', enroll_len=None, num_tests=None, wav_length=None, target_seglen=200, target_seghop=0.5, suffix=''):
        if num_tests is None:
            num_tests = self.df.shape[0]
        if wav_length is not None:
            wav_length = int(self.fs * wav_length)
            self.griffinlim.length = wav_length
        if self.df.shape[0] < num_tests:
            raise ValueError(f"Number of tests {num_tests} exceeds the number of test data {self.df.shape[0]}!")
        self.enroll_len = enroll_len

        if self.config.dataset.target_utter is not None and len(self.config.dataset.target_utter) == 2:
            test_list = self.config.dataset.target_utter[-1]
        else:
            test_list = list(range(num_tests))
        for test_id in test_list:
            target_path, target_offset, target_dur, fs = self.df.loc[test_id]
            # naming
            target_name = os.path.basename(target_path).split('-')[1].rsplit('.', 1)[0]
            out_basename = f'reconstruct-{target_name}-{suffix}'
            target_sig, raw_sr = sf.read(os.path.join(self.config.path.data_prefix, target_path), start=target_offset, stop=target_offset + target_dur)
            if raw_sr != self.fs:
                target_sig = librosa.core.resample(target_sig, raw_sr, self.fs)
            target_sig = torch.from_numpy(target_sig / np.max(np.abs(target_sig))).to(self.config.general.device).float().unsqueeze(0)
            _, target_sig_feat, _ = self.transform(target_sig)
            if mode == 'enroll':
                if enroll_len is not None:
                    enroll_len = int(self.fs * self.enroll_len)
                else:
                    enroll_len = target_dur
                enroll_sig, raw_sr = sf.read(os.path.join(self.config.path.data_prefix, target_path), start=0, stop=enroll_len)
                if raw_sr != self.fs:
                    enroll_sig = librosa.core.resample(enroll_sig, raw_sr, self.fs)
                enroll_sig = torch.from_numpy(enroll_sig / np.max(np.abs(enroll_sig))).to(self.config.general.device).float().unsqueeze(0)
                _, enroll_sig_feat, _ = self.transform(enroll_sig)
                target_rec_spec_enroll = self.rec_inference_one_utterance(enroll_sig_feat, target_sig_feat, target_seglen=target_seglen, target_seghop=target_seghop)
                fig_out_enroll = PlotSpectrogram(target_rec_spec_enroll, self.fs, self.hop_len)
                fig_out_enroll.savefig(os.path.join(self.config.path.abs_out_dir, 'spectrogram', f"Enroll-{out_basename}.png"), dpi=300)
            target_rec_spec_oracle = self.rec_inference_one_utterance(target_sig_feat, target_sig_feat, target_seglen=target_seglen, target_seghop=target_seghop)
            fig_target = PlotSpectrogram(target_sig_feat**0.5, self.fs, self.hop_len)
            fig_out_oracle = PlotSpectrogram(target_rec_spec_oracle, self.fs, self.hop_len)
            fig_target.savefig(os.path.join(self.config.path.abs_out_dir, 'spectrogram', f"target-{target_name}.png"), dpi=300)
            fig_out_oracle.savefig(os.path.join(self.config.path.abs_out_dir, 'spectrogram', f"Oracle-{out_basename}.png"), dpi=300)

    def __call__(self, config, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.rec_inference(**config)


class TSE(Inference):
    def __init__(self, config, suffix=''):
        super(TSE, self).__init__(config)
        self.config.path.abs_out_dir = os.path.join(self.config.general.prefix, self.config.path.out_dir, suffix)
        self.d_embedding = self.config.NetInput.n_embedding
        self.bss_flag = False
        self.ret_flag = False
        if self.config.tse_config['ret_mode'] != 'concate':
            os.makedirs(self.config.path.abs_out_dir, exist_ok=True)
            os.system(f"cp {os.path.join(os.path.dirname(__file__), 'config.py')} {os.path.join(self.config.path.abs_out_dir, 'Config.py')}")

            if self.config.tse_config['ret_mode'] == 'all':
                self.ret = {'MixturePath': [], 'SepBaseName': [],
                            'SDROri': [], 'SDR': [],
                            'SIROri': [], 'SIR': [],
                            'SAROri': [], 'SAR': [],
                            'PESQOri': [], 'PESQ': [],
                            'STOIOri': [], 'STOI': [],
                            'Perm': [], 'Alignment': []}
            elif self.config.tse_config['ret_mode'] == 'target':
                self.ret = {'MixturePath': [], 'SepBaseName': [],
                            'SDROri': [], 'SDR': [],
                            'SIROri': [], 'SIR': [],
                            'SAROri': [], 'SAR': [],
                            'PESQOri': [], 'PESQ': [],
                            'STOIOri': [], 'STOI': [],
                            'Alignment': []}
            self.bss_flag = True
        elif self.config.tse_config['ret_mode'] == 'concate':
            self.ret_flag = True

    def parse_data(self):
        excel_reader = pd.ExcelFile(os.path.join(self.config.general.prefix, self.config.path.data))
        sheet_name_list = excel_reader.sheet_names
        if 'RIRidx' in sheet_name_list:
            df = excel_reader.parse(sheet_name='RIRidx', header=0, usecols=['RIRidx'])
            self.rir_list = list(df['RIRidx'])
            sheet_name_list.remove('RIRidx')
        else:
            raise ValueError(f"RIR index not specified in {self.config.path.data}")
        if 'SIR' in sheet_name_list:
            df = excel_reader.parse(sheet_name='SIR', header=0, usecols=['SIR'])
            self.sir_list = list(df['SIR'])
            sheet_name_list.remove('SIR')
        else:
            self.sir_list = None
        self.src_num = len(sheet_name_list)
        self.df = []
        for i in range(self.src_num):
            self.df.append(excel_reader.parse(sheet_name=sheet_name_list[i], header=0, usecols=self.config.dataset.usecols))
        self.fs = self.df[0].at[0, 'Sr']
        self.config.sigproc.sr = self.fs

    def generate_mixture(self):
        # TODO: when the src number is larger than 2
        if self.rir_mode == 'simulated':
            rir_list = self.idx2rir[self.rir_idx][1]
        elif self.rir_mode == 'real':
            rir_paths = self.idx2rir[self.rir_idx]['path']
            channel = self.idx2rir[self.rir_idx]['channel']
            rir_list = []
            for rir_id in range(len(rir_paths)):
                rir = sio.loadmat(os.path.join(self.config.general.prefix, rir_paths[rir_id]))
                rir = rir['impulse_response'][:, channel].T
                rir = librosa.core.resample(np.asfortranarray(rir), 48000, self.fs)
                rir = rir[:, : int(0.8 * self.fs)]
                #(n_channel,n_sample)
                rir_list.append([rir[i, :] for i in range(self.src_num)])
            rir_list = list(zip(*rir_list))
        self.mic_num = len(rir_list)
        ref_mic = [convolve(self.src_sigs[i, :], rir_list[0][i]) for i in range(self.src_num)]
        interference_sig = np.sum(np.stack(ref_mic[1:], axis=0), axis=0)
        scale_fac = np.std(ref_mic[0]) * 10**(-self.mix_sir / 20) / np.std(interference_sig)
        self.src_sigs[1, :] = scale_fac * self.src_sigs[1, :]
        self.mix_sigs = []
        for mic_id in range(self.mic_num):
            mix = np.sum(np.stack([convolve(self.src_sigs[i, :], rir_list[mic_id][i]) for i in range(self.src_num)]), axis=0)
            self.mix_sigs.append(mix)
        self.mix_sigs = np.stack(self.mix_sigs, axis=0)
        self.mix_sigs = self.mix_sigs[:, : self.src_sigs.shape[1]]
        self.mix_sigs = self.mix_sigs / np.max(np.abs(self.mix_sigs))
        sf.write(self.mix_path, self.mix_sigs.T, self.fs)

    def target_speech_extraction(self, num_tests=None, rir_path=None, rir_mode=None, bss_method=None, ret_mode='all',
                                 target_seglen=100, target_seghop=0.5, enroll_len=30, enroll_mode='all', enrollutt_mode='enroll',
                                 mvae_config=None, ilrma_config=None, ilrma_ori_config=None):
        if num_tests is None:
            num_tests = self.df[0].shape[0]
        if self.df[0].shape[0] < num_tests:
            raise ValueError(f"Number of tests {num_tests} exceeds the number of test data {self.df[0].shape[0]}!")
        with open(os.path.join(self.config.general.prefix, rir_path), 'rb') as f_idx2rir:
            self.idx2rir = pickle.load(f_idx2rir)
        self.rir_mode = rir_mode
        self.enroll_len = int(self.fs * enroll_len)
        if bss_method.startswith('MVAE'):
            os.system(f"cp {os.path.join(os.path.dirname(__file__), 'BSSAlgorithm/mvae.py')} {os.path.join(self.config.path.abs_out_dir, 'mvae.py')}")
        if bss_method == 'MVAE_onehot':
            keymap_dict = {}
            col_names = ['SpkrID', 'SpkrID-original']
            keymap_path = os.path.join(self.config.general.prefix, 'PROJECT/CVAE_training/EsEc_structure/data/train/train-dev-set/clean_data_training/train-set/Egs.clean.100spkr-max.train.0.csv')
            df = pd.read_csv(keymap_path, header=0, index_col=None, usecols=col_names)
            keymap = list(zip(df[col_names[0]], df[col_names[1]]))
            keymap = set(keymap)
            for train_id, real_id in keymap:
                keymap_dict[real_id] = train_id
        time_all = 0
        for test_id in range(num_tests):    
            self.rir_idx = self.rir_list[test_id]
            self.mix_sir = self.sir_list[test_id] if self.sir_list is not None else 0.0

            self.order_list = list(range(self.src_num))
            self.mix_src_basename = [os.path.basename(list(self.df[i].loc[test_id])[0]).split('-')[1].rsplit('.', 1)[0] for i in self.order_list]
            self.mix_src_basename = f'spkr-{self.mix_src_basename[0]}_spkr-' + '-'.join(self.mix_src_basename[1:])
            self.src_path = os.path.join(os.path.dirname(self.config.path.abs_out_dir), f"{self.mix_src_basename}_src.wav")
            self.mix_path = os.path.join(os.path.dirname(self.config.path.abs_out_dir), f"{self.mix_src_basename}_mix.wav")
            # read and write source signals
            if os.path.exists(self.src_path):
                self.src_sigs, _ = sf.read(self.src_path)
                self.src_sigs = self.src_sigs.T
            else:
                DUR = []
                src_sigs_tmp = []
                self.src_sigs = []
                for idx, src_id in enumerate(self.order_list):
                    utt_path, utt_offset, utt_dur, _ = self.df[src_id].loc[test_id]
                    DUR.append(utt_dur)
                    src_sig, raw_sr = sf.read(os.path.join(self.config.general.data_prefix, utt_path), start=utt_offset, stop=utt_offset + utt_dur)
                    if raw_sr != self.fs:
                        src_sig = librosa.core.resample(src_sig, raw_sr, self.fs)
                    src_sigs_tmp.append(src_sig)
                length = np.min(DUR)
                for idx, src_id in enumerate(self.order_list):
                    src_sig = src_sigs_tmp[idx][0:length]
                    src_sig = src_sig - np.mean(src_sig)
                    src_sig = src_sig / np.max(np.abs(src_sig))
                    if idx >= 1:
                        src_sig = src_sig * np.std(self.src_sigs[0]) / np.std(src_sig)
                    self.src_sigs.append(src_sig)
                self.src_sigs = np.stack(self.src_sigs, axis=0)
                sf.write(self.src_path, self.src_sigs.T, self.fs)

            # create mixed signals
            if os.path.exists(self.mix_path):
                self.mix_sigs, _ = sf.read(self.mix_path)
                self.mix_sigs = self.mix_sigs.T
            else:
                self.generate_mixture()
            met_before = mir_eval.separation.bss_eval_sources(self.src_sigs, np.stack([self.mix_sigs[0, :] for _ in range(self.src_num)], axis=0))
            stoi_before = [stoi(self.src_sigs[i, :], self.mix_sigs[0, :], self.fs) for i in range(self.src_num)]
            pesq_before = [pesq(self.src_sigs[i, :], self.mix_sigs[0, :], fs=self.fs) for i in range(self.src_num)]

            for target_id in range(self.src_num):
                self.order_list_new = [self.order_list[target_id]] + self.order_list[: target_id] + self.order_list[target_id + 1:]
                self.sep_basename_list = [os.path.basename(list(self.df[i].loc[test_id])[0]).split('-')[1].rsplit('.', 1)[0] for i in self.order_list_new]
                self.sep_basename = f'target-{self.sep_basename_list[0]}_inferences-' + '-'.join(self.sep_basename_list[1:])
                self.sep_path = os.path.join(self.config.path.abs_out_dir, f"{self.sep_basename}_sep.wav")

                if os.path.exists(self.sep_path):
                    self.sep_sigs, _ = sf.read(self.sep_path)
                    self.sep_sigs = self.sep_sigs.T
                else:
                    # read enrollment signals
                    if enroll_mode == 'None':
                        spkr_embd = torch.ones((0, self.d_embedding)).to(self.config.general.device)
                    else:
                        self.enroll_sigs = []
                        enroll_num = 1 if enroll_mode == 'target' else self.src_num
                        for idx in range(enroll_num):
                            if enrollutt_mode == 'enroll':
                                enroll_path, _, _, _ = self.df[self.order_list_new[idx]].loc[test_id]
                                enroll_sig, raw_sr = sf.read(os.path.join(self.config.general.data_prefix, enroll_path), start=0, stop=self.enroll_len)
                                if raw_sr != self.fs:
                                    enroll_sig = librosa.core.resample(enroll_sig, raw_sr, self.fs)
                                enroll_sig = enroll_sig - np.mean(enroll_sig)
                                enroll_sig = enroll_sig / np.max(np.abs(enroll_sig))
                                self.enroll_sigs.append(enroll_sig)
                                # enroll_sig, raw_sr = sf.read(os.path.join(self.config.general.data_prefix, enroll_path), start=0, stop=60*self.fs)
                                # if raw_sr != self.fs:
                                #     enroll_sig = librosa.core.resample(enroll_sig, raw_sr, self.fs)
                                # enroll_sig = enroll_sig - np.mean(enroll_sig)
                                # enroll_sig = enroll_sig / np.max(np.abs(enroll_sig))
                                # enroll_trim_sig = []
                                # intervals = librosa.effects.split(enroll_sig, top_db=30)
                                # for interval in intervals:
                                #     enroll_trim_sig.append(enroll_sig[interval[0]: interval[1]])
                                # enroll_trim_sig = np.concatenate(enroll_trim_sig, axis=0)
                                # length = min(enroll_trim_sig.shape[0], self.enroll_len)
                                # if length < self.enroll_len:
                                #     print("ENROLLMENT TOO SHORT")
                                # self.enroll_sigs.append(enroll_trim_sig[0: self.enroll_len])
                            elif enrollutt_mode == 'source':
                                self.enroll_sigs.append(self.src_sigs[self.order_list_new[idx], :])
                        enroll_sig = torch.from_numpy(np.stack(self.enroll_sigs, axis=0)).to(self.config.general.device).float()
                        _, enroll_feat, _ = self.transform(enroll_sig)
                        # generate spkr_embd
                        with torch.no_grad():
                            spkr_embd = self.model.get_speaker_embeddings(enroll_feat, target_seglen=target_seglen, target_seghop=target_seghop)
                    # BSS
                    self.mix_spec = np.stack([
                        librosa.core.stft(np.asfortranarray(self.mix_sigs[ch, :]), n_fft=self.stft_len, hop_length=self.hop_len)
                        for ch in range(self.mix_sigs.shape[0])
                    ], axis=1)
                    if bss_method == 'ILRMA_ORI':
                        sep_spec, flag = ilrma(self.mix_spec, **ilrma_config)
                    elif bss_method == 'MVAE':
                        from mvae_standard import Separation
                        nfft = 4096
                        hopsize = nfft // 2
                        def zero_pad(x, hopsize):
                            nframe = x.shape[1]//hopsize + 1
                            if nframe % 4 == 0:
                                return x
                            elif nframe % 4 == 1:
                                return np.pad(x, ((0, 0), (0, 3*hopsize)), mode='constant')
                            elif nframe % 4 == 2:
                                return np.pad(x, ((0, 0), (0, 2*hopsize)), mode='constant')
                            elif nframe % 4 == 3:
                                return np.pad(x, ((0, 0), (0, hopsize)), mode='constant')

                        self.mix_sigs = zero_pad(self.mix_sigs, hopsize)
                        self.mix_spec = np.stack([
                            librosa.core.stft(np.asfortranarray(self.mix_sigs[ch, :]), n_fft=nfft, hop_length=hopsize)
                            for ch in range(self.mix_sigs.shape[0])
                        ], axis=1)
                        # X1 = librosa.stft(self.mix_sigs[0, :].copy(), nfft, hopsize)
                        # X2 = librosa.stft(self.mix_sigs[1, :].copy(), nfft, hopsize)
                        # self.mix_spec = np.stack((X1, X2), axis=1)
                        separator = Separation("/data/hdd0/leleliao/dnn/vae/model/mvae.pt", device=self.config.general.device)
                        sep_spec = separator.separate(self.mix_spec)

                    elif bss_method == 'IN-MVAE-hard':
                        from BSSAlgorithm.in_mvae_hard import MVAEIVA
                        sep_spec, flag = MVAEIVA(self.mix_spec, spkr_embd, self.model,
                                                 device=self.config.general.device, **mvae_config)
                    elif bss_method == 'IN-MVAE-soft':
                        from BSSAlgorithm.in_mvae_soft import MVAEIVA
                        sep_spec, time_per = MVAEIVA(self.mix_spec, spkr_embd, self.model,
                                                 device=self.config.general.device, **mvae_config)
                    elif bss_method == 'AuxIVA':
                        from BSSAlgorithm.myauxiva import auxiva
                        sep_spec = auxiva(self.mix_spec, n_iter=100)
                    time_all += time_per
                    if bss_method == 'MVAE_onehot':
                        src_len = int((self.mix_spec.shape[-1] - 1) * self.hop_len)
                    else:
                        src_len = self.src_sigs.shape[1]
                    self.src_sigs = self.src_sigs[:, :src_len]

                    if bss_method == 'MVAE':
                        self.sep_sigs = np.stack([
                            librosa.core.istft(sep_spec[:, ch, :], hop_length=hopsize, length=src_len)
                            for ch in range(sep_spec.shape[1])
                        ], axis=0)
                    else:
                        self.sep_sigs = np.stack([
                            librosa.core.istft(sep_spec[:, ch, :], hop_length=self.hop_len, length=src_len)
                            for ch in range(sep_spec.shape[1])
                        ], axis=0)
                    self.sep_sigs = self.sep_sigs/np.max(np.abs(self.sep_sigs),1)[:,None]
                    if (bss_method == 'MVAE' or bss_method == 'ILRMA_ORI' or bss_method == 'AuxIVA'):
                        self.sep_sigs = probability(enroll_sig, self.sep_sigs, xvecmodel_path=self.config.plda.xvecmodel_path, \
                            xvecmean_path=self.config.plda.xvecmean_path, lda_path=self.config.plda.lda_path, plda_path=self.config.plda.plda_path)
                    sf.write(self.sep_path, self.sep_sigs.T, self.fs)
                    

                # objective metrics
                if ret_mode == 'all':
                    self.ret['MixturePath'].extend([os.path.join(os.path.dirname(self.config.path.out_dir), f"{self.mix_src_basename}_mix.wav")] * 2)
                    self.ret['SepBaseName'].extend([f"{self.sep_basename}_sep.wav"] * 2)

                    met = mir_eval.separation.bss_eval_sources(self.src_sigs, self.sep_sigs)
                    for i, m in enumerate(['SDR', 'SIR', 'SAR']):
                        self.ret[m].extend(met[i].tolist())
                        self.ret[m + 'Ori'].extend(met_before[i].tolist())
                    self.ret['Perm'].extend(met[-1].tolist())
                    if bss_method == 'MVAE_onehot' and labels[0] == labels[1]:
                        self.ret['Alignment'].extend([False] * self.src_num)
                    else:
                        self.ret['Alignment'].extend([True if met[-1][i] == self.order_list_new[i] else False for i in range(self.src_num)])
                        # self.ret['Alignment'].extend([True if ((target_id == 0 and flag == 0) or (target_id == 1 and flag == 1)) else False for _ in range(self.src_num)])
                    print(f"{test_id}: {met}")
                    print(self.ret['Alignment'][-1])

                    stoi_after = [stoi(self.src_sigs[i, :], self.sep_sigs[met[-1][i], :], self.fs) for i in range(self.src_num)]
                    pesq_after = [pesq(self.src_sigs[i, :], self.sep_sigs[met[-1][i], :], fs=self.fs) for i in range(self.src_num)]
                    self.ret['PESQOri'].extend(pesq_before)
                    self.ret['STOIOri'].extend(stoi_before)
                    self.ret['PESQ'].extend(pesq_after)
                    self.ret['STOI'].extend(stoi_after)
                elif ret_mode == 'target':
                    self.ret['MixturePath'].append(os.path.join(os.path.dirname(self.config.path.out_dir), f"{self.mix_src_basename}_mix.wav"))
                    self.ret['SepBaseName'].append(f"{self.sep_basename}_sep.wav")

                    met_target = mir_eval.separation.bss_eval_sources(self.src_sigs, np.stack([self.sep_sigs[0, :] for _ in range(self.src_num)], axis=0))
                    met = mir_eval.separation.bss_eval_sources(self.src_sigs, self.sep_sigs)
                    for i, m in enumerate(['SDR', 'SIR', 'SAR']):
                        self.ret[m].append(met_target[i][target_id])
                        self.ret[m + 'Ori'].append(met_before[i][target_id])
                    self.ret['Alignment'].append(True if met[-1][target_id] == 0 else False)
                    print(f"{test_id}: {met_target[1][target_id]}")

                    stoi_after = stoi(self.src_sigs[target_id, :], self.sep_sigs[0, :], self.fs)
                    pesq_after = pesq(self.src_sigs[target_id, :], self.sep_sigs[0, :], fs=self.fs)
                    self.ret['PESQOri'].append(pesq_before[target_id])
                    self.ret['STOIOri'].append(stoi_before[target_id])
                    self.ret['PESQ'].append(pesq_after)
                    self.ret['STOI'].append(stoi_after)
        
        print(f"time: {time_all}")
        self.ret_path_xls = os.path.join(self.config.general.prefix, self.config.path.abs_out_dir, f'Result_{ret_mode}.xls')
        self.ret_path_pic = os.path.join(self.config.general.prefix, self.config.path.abs_out_dir, f'Result_{ret_mode}.pkl')
        df = pd.DataFrame(self.ret)
        excel_writer = pd.ExcelWriter(self.ret_path_xls)
        df.to_excel(excel_writer, index_label="Index")
        excel_writer.save()
        excel_writer.close()
        with open(self.ret_path_pic, 'wb') as f:
            pickle.dump(self.ret, f)

    def concate_ret(self, state=0):
        mode = ['FM']
        # mode = ['FF']
        # mode = ['MM']
        root_path = 'PROJECT/CVAE_training/EsEc_structure/inference_results/tse/IN-MVAE-soft'
        # root_path = 'PROJECT/CVAE_training/EsEc_structure/inference_results_stage2/tse/IN-MVAE-soft'
        if state == 0:
            #保存成xls和pkl格式
            # mode = ['FM', 'FF', 'MM']
            # method = [
            #     # 'ILRMA_PRA',
            #     'MVAE_onehot-bpmle-epoch=500',
            #     'MVAE_alternate_align_W-bpmle-s1_epoch=500',
            #     'MVAE_ilrmaW30-bpmle-epoch=8-enroll=30',
            # ]
            
            method = ['sub_epoch=250.ret_mode_all.enroll_mode_all.ilrma_init_True.bp_encoderinit_l1',]
            # method = ['sub_epoch=1600.ret_mode_all.enroll_mode_all.ilrma_init_True.bp_encoderinit_l1',]
            out_path_xls = os.path.join(self.config.general.prefix, root_path, f'concate_ret--{mode}.xls')
            out_path_pkl = os.path.join(self.config.general.prefix, root_path, f'concate_ret--{mode}.pkl')
            cat_keys_metric = ['SIR', 'SDR', 'SAR', 'PESQ', 'STOI']
            cat_keys_accuracy = ['Correct', 'Total', 'Accuracy']
            cat_ret = defaultdict(list)
            
            for meth, mod in product(method, mode):
                path = os.path.join(self.config.general.prefix, root_path, mod, meth, 'Result_all.pkl')
                cat_ret['path'].append(path)
                with open(path, 'rb') as f:
                    ret = pickle.load(f)
                for key in cat_keys_metric:
                    cat_ret[key].append(stat.mean(ret[key]))
                    cat_ret[key + '-imp'].append(stat.mean([ret[key][i] - ret[key + 'Ori'][i] for i in range(len(ret[key]))]))
                total_num = len(ret['Alignment'])
                correct_num = sum(ret['Alignment'])
                accuracy = correct_num / total_num
                cat_ret['Correct'].append(correct_num)
                cat_ret['Total'].append(total_num)
                cat_ret['Accuracy'].append(accuracy)
            df = pd.DataFrame(cat_ret)
            excel_writer = pd.ExcelWriter(out_path_xls)
            df.to_excel(excel_writer, index_label="Index")
            excel_writer.save()
            excel_writer.close()
            with open(out_path_pkl, 'wb') as f:
                pickle.dump(cat_ret, f)
            """ value in dict """
            """ 
            [meth-1_mode-1, meth-1_mode-2, ..., meth-N_mode-1, ...]
            """
        
        elif state == 1:
            key_list = ['SIR-imp', 'SDR-imp', 'PESQ', 'STOI', 'Correct', 'Accuracy']
            ret_path = os.path.join(self.config.general.prefix, root_path, f'concate_ret--{mode}.pkl')
            with open(ret_path, 'rb') as f:
                ret_dict = pickle.load(f)
            for key in key_list:
                mat_path = os.path.join(self.config.general.prefix, root_path, f"{key}--{mode}.mat")
                ret = np.stack(np.split(np.array(ret_dict[key]), len(ret_dict[key]) / 3), 1)
                sio.savemat(mat_path, {'res': ret})

    def __call__(self, config, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        if self.bss_flag:
            self.target_speech_extraction(**config)
        elif self.ret_flag:
            self.concate_ret(state=0)


class ViewEmbedding(Inference):
    
    def __init__(self, config, suffix=''):
        super(ViewEmbedding, self).__init__(config)
        self.config.path.abs_out_dir = os.path.join(self.config.path.common_prefix, self.config.path.model, suffix)
        self.d_embedding = self.config.NetInput.n_embedding
        self.transform_new = Transform(self.config.sigproc)
        os.makedirs(self.config.path.abs_out_dir, exist_ok=True)
        os.system(f"cp {os.path.join(os.path.dirname(__file__), 'config.py')} {os.path.join(self.config.path.abs_out_dir, 'Config.py')}")

    def parse_data(self):
        test_dataset = NoisyDataset(os.path.join(self.config.path.common_prefix, self.config.path.data), self.config.dataset.usecols_ark,
                                    self.config.dataset.usecols_withspkrid, self.config.path.common_prefix, self.config.path.data_prefix, extra_func=None)
        self.test_loader = DataLoader(test_dataset, batch_size=64, num_workers=0, collate_fn=CollateFnWrapper, drop_last=False, shuffle=False)

    def generate_embeddings(self):
        pass
        # writer = SummaryWriter(self.config.path.abs_out_dir)
        # for in_data in self.test_loader:
        #     data_target = in_data.data_target.to(self.cfgs.general.device)
        #     data_noise = in_data.data_noise.to(self.cfgs.general.device)
        #     mix_info = in_data.mixinfo
        #     remark = in_data.remark
        #     data_target, data_noisy = self.transform_new(remark, data_target, data_noise, mix_info)
        #     spkr_embd = self.model.get_speaker_embeddings(data_target)


def Infer(config, seed):
    if config.info.mode == 'vc':
        config.path.out_dir = os.path.join(config.path.out_dir, 'vc')
        config.path.data = 'DATASET/LibriSpeech/inference/Pair.clean.100spkr-max.test.2src-MM-40*1.t60-36ms.xls'
        # config.path.data = 'DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/test/Egs.train.stage2.1.xls'
        suffix = config.path.model.rsplit('.', 1)[0].rsplit('/', 1)[1].split('--', 1)[1]
        # suffix = 'test-01/FM/epoch=14'
        vc_ins = VoiceConversion(config, suffix=suffix)
        vc_ins(config.vc_config, seed)
    elif config.info.mode == 'rec':
        config.path.out_dir = os.path.join(config.path.out_dir, 'reconstruct')
        config.path.data = 'DATASET/LibriSpeech/inference/Pair.clean.100spkr-max.test.2src-FM-40*1.t60-36ms.xls'
        suffix = config.path.model.rsplit('.', 1)[0].rsplit('/', 1)[1].split('--', 1)[1]
        # suffix = 'test-01/FM/'
        rec_ins = ReconstructSpectrogram(config, suffix=suffix)
        rec_ins(config.rec_config, seed)
    elif config.info.mode == 'noisyrec':
        config.path.out_dir = os.path.join(config.path.out_dir, 'noisy reconstruct')
        config.path.data = 'DATASET/LibriSpeech/inference/Pair.clean.100spkr-max.test.2src-FM-40*1.t60-36ms.xls'
        # config.path.data = 'DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/test/Egs.train.stage2.2.xls'
        suffix = config.path.model.rsplit('.', 1)[0].rsplit('/', 1)[1].split('--', 1)[1]
        # suffix = 'test-01/FM/epoch=10'
        rec_ins = NoisyReconstruct(config, suffix=suffix)
        rec_ins(config.noisyrec_config, seed)
    elif config.info.mode == 'tse':
        config.path.out_dir = os.path.join(config.path.out_dir, 'tse', config.tse_config.bss_method, 'FM')
        config.path.data = 'DATASET/LibriSpeech/inference/Pair.clean.100spkr-max.test.2src-FM-40_1.t60-36ms.xls'
        # config.path.data = 'DATASET/LibriSpeech/inference/Pair.clean.40spkr-max.2src-FM-40_1.t60-36ms.xls'
        suffix_epoch = config.path.model.rsplit('.', 1)[0].rsplit('/', 1)[1].split('--', 1)[1]
        suffix_ret_mode = 'ret_mode_' + config.tse_config.ret_mode
        suffix_enroll_mode = 'enroll_mode_' + config.tse_config.enroll_mode
        suffix_ilrma_init = 'ilrma_init_' + str(config.tse_config.mvae_config['ilrma_init'])
        suffix_latent_meth = config.tse_config.mvae_config['latent_meth']
        suffix = suffix_epoch+'.'+suffix_ret_mode+'.'+suffix_enroll_mode+'.'+suffix_ilrma_init+'.'+suffix_latent_meth
        # suffix = 'test-t60-36/unseen_data/FF/MVAE_ilrmaW30-bpmle-epoch=8-enroll=30'
        tse_ins = TSE(config, suffix=suffix)
        tse_ins(config.tse_config, seed)



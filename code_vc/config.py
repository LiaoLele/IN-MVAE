from PreProcess.utils import myDotDict
from scipy import stats
import numpy as np
import random
import torch
import os


def CreateMixWeight():
    ncomponent = 10
    mixture_weight_list = random.choices(np.linspace(0.1, 0.5, num=5, endpoint=True), k=ncomponent)  # list(np.random.uniform(low=0.4, high=0.5, size=(ncomponent)))  #   # 
    mixture_weight_list = mixture_weight_list / sum(mixture_weight_list)
    x_range = np.linspace(0, 20, num=512, endpoint=True)
    loc_list = random.sample(list(x_range[129:]), k=ncomponent)
    cum_weights = [30, 40, 50, 60, 65, 70, 75, 80, 85]
    df_list = [1] + random.choices(range(1, 10), cum_weights=cum_weights, k=ncomponent - 1)
    peak = np.random.uniform(low=0.7, high=0.9, size=(1))[0]
    y = []
    for i in range(ncomponent):
        df = df_list[i]
        loc = loc_list[i]
        mixture_weight = mixture_weight_list[i]
        y.append(mixture_weight * stats.t.pdf(x_range, df, loc))
        # To shift and/or scale the distribution use the loc and scale parameters
    y = sum(y)
    y = y / np.max(y) * peak
    return y


def CreateFreqWeight():
    weight = torch.cat([torch.full((128,), 70 / 128), torch.full((513 - 128, ), (513 - 70) / (513 - 128))])
    return weight


class cfgs():

    pre_log = [
        # 'zloss = torch.mean(torch.log10(zloss + 0.1))',
        # 'zloss = torch.mean(torch.log10(zloss))',
        # 'return torch.relu(zloss - 1.69), zloss '
    ]

    general = {
        # 'train': True,
        # 'inference': False,
        'train': False,
        'inference': True,
        'prefix': '/data/hdd0/leleliao/',
        'data_prefix': ['/data/ssd1/zhaoyi.gu', '/data/hdd0/zhaoyi..gu'][0],
        'device': torch.device(0),
        'model_type': 'cnn',
    }
    general = myDotDict(general)

    NetInput = {
        'chunk_size': 64,
        'hop_size': 32,
        'n_embedding': 128,
    }
    NetInput = myDotDict(NetInput)

    SpeakerEncoder = {
        # 'c_in': 513,
        'c_in': 512,
        'c_out': 128,
        'n_conv_blocks': 6,
        'n_dense_blocks': 6,
        'conv_bank_scale': 1,
        'max_bank_width': 8,
        'c_bank': 128,
        'c_h': 128,
        'kernel_size': 5,
        'stride_list': [1, 2, 1, 2, 1, 2],
        'activation_layer': 'nn.ReLU()',
        'dropout_rate': 0.0,
        'batchnorm': False,
    }
    SpeakerEncoder = myDotDict(SpeakerEncoder)

    ContentEncoder = {
        # 'c_in': 513,
        'c_in': 512,
        'c_out': 64,
        'n_conv_blocks': 4,
        'conv_bank_scale': 1,
        'max_bank_width': 8,
        'c_bank': 128,
        # 'c_h': [512, 256, 128, 64, 32],
        'c_h': [512, 512, 256, 128, 64],
        # 'c_h': [513, 513, 256, 128, 64],
        'kernel_size': 5,
        'stride_list': [1, 2, 1, 2], 
        'activation_layer': 'nn.ReLU()',
        'dropout_rate': 0.0,
        # 'mu_act_param': {
        #     'act': 'tanh',
        #     'params': {
        #         'alpha': 0.005,
        #     }
        # },
        # 'var_act_param': {
        #     'act': 'relu',
        #     'params': {
        #         'alpha': 1.,
        #     }
        # }
    }
    ContentEncoder = myDotDict(ContentEncoder)

    Decoder = {
        'c_cont': 64,
        'c_spkr': 128,
        'c_out': 512,
        'n_conv_blocks': 4,
        # 'c_h': [64, 64, 128, 256, 513],
        'c_h': [64, 64, 128, 256, 512],
        # 'c_h': [32, 64, 128, 256, 512],
        'kernel_size': 5,
        'upsamp_list': [2, 1, 2, 1],
        'activation_layer': 'nn.ReLU()',
        'dropout_rate': 0.0,
    }
    Decoder = myDotDict(Decoder)

    sigproc = {
        'sr': 16000,
        'stft_len': 0.128, # 0.128
        'hop_len': 0.032, # 0.032
        'nmels': 512,
        'nmfcc': None,
        # 'calc_spec': True,
        # 'spec_mode': 'power',
        'calc_spec': False,
        'calc_mel': True,
        'mel_mode': 'db',
    }
    sigproc = myDotDict(sigproc)

    if general.train is True:
        info = {
            'mode': 'train',   # ['train', 'checkdata']
            'stage': 2, 
            'activate_speaker_encoder': False,
            'activate_content_encoder': True,
            'activate_decoder': True,
        }
        # lambda_zbackloss = myDotDict(info)
        info = myDotDict(info)

        path = {
            'train': [[os.path.join(general['prefix'], 'DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/train/Egs.train.stage1.csv')],\
                os.path.join(general['prefix'], 'DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/train/Egs.train.stage2.ARKINFO.csv')][1],
            'dev': [[os.path.join(general['prefix'], 'DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/dev/Egs.train.stage1.csv')],\
                os.path.join(general['prefix'], 'DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/dev/Egs.train.stage2.ARKINFO.csv'),\
                [os.path.join(general['prefix'], 'DATASET/LibriSpeech/dev/ampnorm_meannorm_trim30/segments/Egs.dev.stage1.csv')]][1],
            'test':[[os.path.join(general['prefix'], 'DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/test/Egs.train.stage1.csv')],\
                os.path.join(general['prefix'], 'DATASET/LibriSpeech/train/ampnorm_meannorm_trim30/segments/test/Egs.train.stage2.ARKINFO.csv'),\
                [os.path.join(general['prefix'], 'DATASET/LibriSpeech/test/ampnorm_meannorm_trim30/segments/Egs.test.stage1.csv')]][1],
            # 'root': os.path.join(general['prefix'], 'PROJECT/CVAE_training/EsEc_structure/train_results/stage1'),
            # 'model': '/data/hdd0/leleliao/PROJECT/CVAE_training/EsEc_structure/train_results/stage1/model',
            'root': os.path.join(general['prefix'], 'PROJECT/CVAE_training/EsEc_structure/train_results_mse_64/stage1'),
            'model': '/data/hdd0/leleliao/PROJECT/CVAE_training/EsEc_structure/train_results_mse_64/stage2/model',

            'stage_one': os.path.join(general['prefix'], 'PROJECT/CVAE_training/EsEc_structure/train_results_mse_64/stage1/model/state_dict--sub_epoch=250.pt'),
            'resume': None,
            # 'resume': os.path.join(general['prefix'], 'PROJECT/CVAE_training/EsEc_structure/train_results_mse_64/stage1/model/resume_model--250.pt'),
            
        }
        path = myDotDict(path)

        dataset = {
            'usecols_egs': ['SpkrID', 'RelativePath', 'Offset', 'Duration', 'Sr'],
            'usecols_ark': ['RelativePath', 'NumberOfEgsPerArk', 'NumberOfSpeakersPerEgs', 'Remark'],
            'CreateMixWeight': lambda: CreateMixWeight(), #得到的是这个函数return的东西

        }
        dataset = myDotDict(dataset)

        dataloader = {
            'batch_size': 64,
            'nbatch_train': 469,
            'nbatch_dev': None,
            'nbatch_test': None,
            'droplast_train': False,
            'droplast_dev': True,
            'droplast_test': True,
            'num_workers': 2,
            'droplast_extra': True,
        }
        dataloader = myDotDict(dataloader)

        opt_strategy = {
            # 'optim_strategy': "torch.optim.Adam(self.model.parameters(), lr=self.cfgs.opt_strategy.lr, weight_decay=self.cfgs.opt_strategy.weight_decay, amsgrad=self.cfgs.opt_strategy.amsgrad)",
            'optim_strategy': "torch.optim.Adam(filter(lambda x: x.requires_grad is True, self.model.parameters()), lr=self.cfgs.opt_strategy.lr, weight_decay=self.cfgs.opt_strategy.weight_decay, amsgrad=self.cfgs.opt_strategy.amsgrad)",
            'lr': 0.0001,
            'weight_decay': 0.001,
            'amsgrad': True,

            'use_lr_decay': False,
            'min_lr': 0.00005,
            'lr_activate_idx': 0,
            'lr_deactivate_idx': 500,

            'dropout_strategy': {
                "self.model.speaker_encoder.dropout_rate = dropout_rate": "0@0.0",  # "0@0,150@0,300@0.15,500@0.02",
                "self.model.content_encoder.dropout_rate = dropout_rate": "0@0.0",  # "0@0,150@0,300@0.15,500@0.02",
                "self.model.decoder.dropout_rate = dropout_rate": "0@0.0",  # "0@0,150@0,300@0.15,500@0.02",
            },

            'use_grad_clip': True,
            # "grad_clip_strategy": "torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)",
            "grad_clip_strategy": "torch.nn.utils.clip_grad_norm_(filter(lambda x: x.requires_grad is True, self.model.parameters()), 3.0)",
        }
        opt_strategy = myDotDict(opt_strategy)

        train = {
            'resume': False,
            'tensorboard_title': 'librispeech',
            'total_epoch_num': 5000,

            'nan_backroll': True,
            'pause_when_nan': False,
            'max_nan_allowance': 3,

            'logging_interval': 50,
            'validate_interval': 469, 
            'save_stdt_interval': 50,
            'save_ckpt_interval': 50,
        }
        train = myDotDict(train)

        loss = {
            'loss_type': 'mse_loss',   # ['basic', 'l1_loss', 'mse_loss']
            'freq_weight': None,
            # 'freq_weight': lambda: CreateFreqWeight(),
            'lambda_rec': 10,
            'lambda_kl': 1,
            'lambda_zbackloss': 0,
            'anneling_iters': 0,
            'include_zgrad_loss': False,
            'include_zmse_loss': False,
            'zmse_loss_config': {
                'bp_num': 12,
                'sampling': True,
                'bp_lr': 2.5e-5,
                'bp_lr_trainable': False,
            },
            'rec_loss_reduce_mode': 'sum',
        }
        loss = myDotDict(loss)

    elif general.inference is True:
        info = {
            'mode': 'tse',
            # 'mode': 'rec',
            # 'mode': 'vc',
            'use_spkr_affine': None,   # used in rec
            'noise_type': 'mix',   # used in noisyrec
            'load_model': True,
        }
        info = myDotDict(info)

        path = {
            'common_prefix': ['/data/hdd0/leleliao/', '/home/user/zhaoyi.gu/mnt/g2/'][0], 
            'data_prefix': ['/data/ssd1/zhaoyi.gu', '/data/hdd0/zhaoyi..gu'][1],
            'model': 'PROJECT/CVAE_training/EsEc_structure/train_results_mse_64/stage1/model/state_dict--sub_epoch=250.pt',
            # 'model': 'PROJECT/CVAE_training/EsEc_structure/train_results_mse_64/stage2/model/state_dict--sub_epoch=1600.pt',
            'data': 'DATASET/LibriSpeech/inference/Pair.clean.100spkr-max.test.2src-FM-40*1.t60-61ms.xls',
            'out_dir': 'PROJECT/CVAE_training/EsEc_structure/inference_results/',
        }
        path = myDotDict(path)

        dataset = {
            'usecols': ['RelativePath', 'Offset', 'Duration', 'Sr'],
            'usecols_withspkrid': ['SpkrID', 'RelativePath', 'Offset', 'Duration', 'Sr'],
            'usecols_ark': ['RelativePath', 'NumberOfEgsPerArk', 'NumberOfSpeakersPerEgs', 'Remark'],
            'CreateMixWeight': lambda: CreateMixWeight(),
            'target_utter': None,
        }
        dataset = myDotDict(dataset)

        vc_config = {
            'num_tests': 10,
            'wav_length': None,
            # 'wav_length': 5,  # second
            'target_seglen': 100,  # frame
            'target_seghop': 0.5  # percentage
        }
        vc_config = myDotDict(vc_config)

        rec_config = {
            'mode': 'enroll',
            'enroll_len': 30,
            'num_tests': None,
            'wav_length': None,
            'target_seglen': 100,  # frame
            'target_seghop': 0.5  # percentage
        }
        rec_config = myDotDict(rec_config)

        noisyrec_config = {
            'mode': 'oracle',
            'enroll_len': 30,
            'num_tests': 20,
            'wav_length': None,
            'target_seglen': 100,  # frame
            'target_seghop': 0.5,  # percentage
        }
        noisyrec_config = myDotDict(noisyrec_config)

        tse_config = {
            'num_tests': 40, # None
            'ret_mode': 'all',  # 'concate','target','all'
            'rir_path': os.path.join(general.prefix, 'DATASET/rir/MIRD/idx2rir.pkl'),
            'rir_mode': 'real',
            'bss_method': 'IN-MVAE-soft', #['AuxIVA', 'ILRMA_ORI', 'MVAE', 'IN-MVAE-hard','IN-MVAE-soft']
            'target_seglen': 100,  # frame
            'target_seghop': 0.5,  # percentage
            'enroll_len': 30,
            'enroll_mode': 'all',  # ['None', 'target', 'all']
            'enrollutt_mode': 'source',  # ['enroll', 'source']
            'mvae_config': {
                'n_iter': 100, # 1000
                'ilrma_init': True,
                'alternate_align_init': False,
                'latent_meth': 'bp_l1',  # ['bp_l1', 'bp_encoderinit_l1', 'bp_mle', 'bp_map', 'encoder', 'bp_encoderinit_mle', 'bp_encoderinit_map', 'encoder_update']
                'nsamp': 1,
            },
            'ilrma_config': {
                'alternate_align_init': False,
                'n_iter': 100,
            },
            # 'auxiva_config': {
            #     'proj_back': True,
            #     'n_iter': 100,
            # },
        }
        tse_config = myDotDict(tse_config)

        plda = {
            'plda_path': '/home/nis/lele.liao/projects/code_vc/PLDA/data/plda_base.0.hdf5',
            'xvecmean_path': '/home/nis/lele.liao/projects/code_vc/PLDA/data/xvec_mean.train.0.pkl',
            'lda_path': '/home/nis/lele.liao/projects/code_vc/PLDA/data/lda_machine.withdimreduction.0.hdf5',
            'xvecmodel_path': '/home/nis/lele.liao/projects/x-vector/xvector_train_result/model/state_dict--sub_epoch=200.pt',
        }
        plda = myDotDict(plda)

        plda_vc = {
            'plda_path': '/home/nis/lele.liao/projects/code_vc/PLDA/data_vc/plda_base.0.hdf5',
            'xvecmean_path': '/home/nis/lele.liao/projects/code_vc/PLDA/data_vc/xvec_mean.train.0.pkl',
            'lda_path': '/home/nis/lele.liao/projects/code_vc/PLDA/data_vc/lda_machine.withdimreduction.0.hdf5',
        }
        plda_vc = myDotDict(plda_vc)

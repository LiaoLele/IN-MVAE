import os
from utils import myDotDict


class cfgs():

    pre_log = [
        "Use dropout strategy: 0@0,100@0.1",
        # "Optim strategy is Adam!",
    ]

    general = {
        'train': True,
        'checkdata': False,
        'prefix': ['/data/hdd0/zhaoyigu/', '/home/user/zhaoyi.gu/mnt/g2/'][0],
        'data_prefix': ['/data/ssd1/zhaoyi.gu', '/data/ssd0/zhaoyi.gu'][0],
        'device': 1,
    }
    general = myDotDict(general)

    path = {
        # 'train': os.path.join(general['prefix'], 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/combine/egs/egs.train.pkl'),
        # 'vad': os.path.join(general['prefix'], 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/DIR_AUG_DATA/combine/egs/egs.vad.pkl'),
        # 'diagnosis': os.path.join(general['prefix'], 'DATASET/SEPARATED_LIBRISPEECH/TRAIN_DATA/AUG_DATA/SEP_AUG_DATA/combine/egs/egs.vad.pkl'),
        # 'root': os.path.join(general['prefix'], 'PROJECT/Xvector_speaker_encoder_data/diraug_librispeech_02/'),
        # 'resume': os.path.join(general['prefix'], 'PROJECT/Xvector_speaker_encoder_data/diraug_librispeech_01/model/resume_model--250.pt'),
        'train': '/data/hdd0/zhaoyigu/DATASET/Librispeech/concatenate/train_clean/egs/egs_train.pkl',
        'vad': '/data/hdd0/zhaoyigu/DATASET/Librispeech/concatenate/train_clean/egs/egs_vad.pkl',
        'diagnosis': '/data/hdd0/zhaoyigu/DATASET/Librispeech/concatenate/train_clean/egs/egs_diagnosis.pkl',
        'root': '/home/nis/lele.liao/projects/x-vector/xvector_train_result/',
        'resume': '/home/nis/lele.liao/projects/x-vector/xvector_train_result/model/resume_model--200.pt',
    }
    path = myDotDict(path)

    sigproc = {
        'sr': 16000,
        'stft_len': 0.064,
        'stft_hop': 0.016,
    }
    sigproc = myDotDict(sigproc)

    model = {
        'feat_num': 30,
        'tdnn_channels': [512, 512, 512, 512, 1500],
        'fc_channels': [512, 512],
    }
    model = myDotDict(model)

    train = {
        'nan_backroll': True,
        'pause_when_nan': False,
        'max_nan_allowance': 3,

        'nbatch_for_validation': 250,
        'batch_size': 64,
        'num_workers': 2,
        'lr': 0.005,
        'total_epoch_num': 5000,
    
        'optim_strategy': "torch.optim.SGD(model.parameters(), lr=cfgs.train.lr, momentum=0.5)",

        'use_lr_decay': True,
        'min_lr': 0.0005,
        'lr_activate_idx': 80,
        'lr_deactivate_idx': 500,

        'use_dropout': True,
        'dropout_strategy': "0@0.0,100@0.1",  # "0@0,150@0,300@0.15,500@0.02",

        'validate_interval': 900,
        'logging_interval': 90,
        'save_statedict_interval': 10,
        'save_resumemodel_interval': 10,
        'resume': True,
        'tensorboard_title': 'librispeech',
        "clip": {'model by norm': 3},
    }
    train = myDotDict(train)
    
    mfcc = {
        'n_fft': int(sigproc['stft_len'] * sigproc['sr']),
        'win_length': int(sigproc['stft_len'] * sigproc['sr']),
        'hop_length': int(sigproc['stft_hop'] * sigproc['sr']),
        'n_mels': model['feat_num'],
    }

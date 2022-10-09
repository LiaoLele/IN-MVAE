import os
import pickle
import numpy as np
import pandas as pd
import soundfile as sf

def ConcateDev():
    datafile = '/data/hdd0/zhaoyigu/PROJECT/CVAE_training/EsEc_structure/data/train/train-dev-set/spk2utt.clean.100spkr-max.dev.pkl'
    outpath = '/data/hdd0/zhaoyigu/PROJECT/CVAE_training/EsEc_structure/data/inference/spk2utt.5s.dev.pkl'
    prefix = '/data/ssd1/zhaoyi.gu/'
    dataoutpath = 'Librispeech/train/ampnorm_meannorm_trim30/ForTest'
    os.makedirs(dataoutpath, exist_ok=True)
    metainfo = {
        'RelativePath': [],
        'LenInSamples': [],
        'SpkrID': [],
        'Sr': [],
        'LenInMin': [],
    }
    fs = 16000
    with open(datafile, 'rb') as f:
        spk2utt = pickle.load(f)
    for key in spk2utt.keys():
        utt = []
        paths, offsets, durs = list(zip(*spk2utt[key]))
        total_dur = sum(durs)
        if total_dur < int(38 * 16000):
            continue
        for path, offset, dur in spk2utt[key]:
            s, _ = sf.read(os.path.join(prefix, path), start=offset, stop=offset + dur)
            utt.append(s)
        utt = np.concatenate(utt)
        assert len(utt) == total_dur
        sf.write(os.path.join(prefix, dataoutpath, os.path.basename(path)), utt, fs)
        spkrID = int(os.path.basename(path).rsplit('-')[1].split('.')[0])
        metainfo['RelativePath'].append(os.path.join(dataoutpath, os.path.basename(path)))
        metainfo['LenInSamples'].append(total_dur)
        metainfo['SpkrID'].append(spkrID)
        metainfo['Sr'].append(fs)
        metainfo['LenInMin'].append(round(total_dur / fs / 60, 2))
    meta = pd.DataFrame(metainfo)
    meta.to_csv(os.path.join(prefix, dataoutpath, 'metainfo.csv'), index_label='Index')




if __name__ == '__main__':
    ConcateDev()
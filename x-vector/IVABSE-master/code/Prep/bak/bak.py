
def get_seped_utter_4src(rirdata_path, pairinfo_path, out_path, sub_state=0, method='ILRMA', prefix=None, rep_metrics_path=None, **kwargs):
    """ separate mixtures in makemix.n.pkl and save separated utterances to out_path """
    """
    Args:
        `rirdata_path`: [str] path where idx2rir.pkl is saved
        `pairinfo_path`: [list] path where makemix.pkl is saved
        `out_path`: [list] path where .wav file will be saved
        `sub_state`: [int] 0 is execution state; 1 is debugging state
        `method`: [str] separation method "ILRMA" and "MVAE"
        `kwargs`: [dict] params for STFT and method
    Out:
        `metric_dict`: [Not returned but saved][dict]
                    dict object that saves SDR, SIR, SAR information of separated data in makemix.n.pkl order
                    {'mix': [makemix[0], makemix[1], ...], 'SDR': {'ori': [], 'sep': []}, 'SIR': {'ori': [], 'sep': []}, 'SAR': {'ori': [], 'sep': []}}
        `metrics.n.pkl`: pickle file that saves metric_dict for job n
        `log.n.txt`: txt file that saves log for job n
                     each line of log is 'Finish processing index m for job n' 
        `sep`: saved in wav file
    """
    print(sub_state)
    f_idx2rir = open(os.path.join(rirdata_path, 'idx2rir.pkl'), 'rb')
    idx2rir = pickle.load(f_idx2rir)
    f_idx2rir.close()
    nfft = int(kwargs['fs'] * kwargs['stft_len']) 
    hop = int(kwargs['fs'] * kwargs['stft_hop'])
    fs = kwargs['fs']

    if method == 'MVAE_onehot':
        modelfile_path = kwargs['vae_model_path']
        n_embedding = kwargs['embedding_dim']
        device = kwargs['device']
        model = vae.net(n_embedding)
        model.load_state_dict(torch.load(modelfile_path, map_location=device))
        model.to(device)
        model.eval()
    elif method == 'MVAE_ge2e':
        fb_mat = torch.from_numpy(librosa.filters.mel(16000, nfft, n_mels=40)).unsqueeze(0)
        vaemodel_path = kwargs['vae_model_path']
        spkrmodel_path = kwargs['spkr_model_path']
        n_embedding = kwargs['embedding_dim']
        device = kwargs['device']
        vae_model = vae.net(n_embedding)
        vae_model.load_state_dict(torch.load(vaemodel_path, map_location=device))
        vae_model.to(device)
        vae_model.eval()
        spkr_model = speaker_encoder_ge2e()
        spkr_model_dict = spkr_model.state_dict()
        pretrained_dict = torch.load(spkrmodel_path, map_location=torch.device('cpu'))
        pretrained_dict_rename = {}
        for k, v in pretrained_dict.items():
            try:
                param_name = k.split('.', 1)[1]
                pretrained_dict_rename[param_name] = v
            except IndexError:
                pass
        pretrained_dict_rename = {k: v for k, v in pretrained_dict_rename.items() if k in spkr_model_dict}
        spkr_model_dict.update(pretrained_dict_rename)
        spkr_model.load_state_dict(spkr_model_dict)
        spkr_model.cuda(device)
        spkr_model.eval()
    
    for path_idx, pairinfo in enumerate(pairinfo_path):
        print('Current makemix is {}'.format(pairinfo))
        os.makedirs(out_path[path_idx], exist_ok=True)
        
        f_makemix = open(pairinfo, 'rb')
        makemix = pickle.load(f_makemix)
        f_makemix.close()
        if sub_state == 0:
            f_metrics = open(os.path.join(out_path[path_idx], 'metrics-{}.pkl'.format(os.path.basename(pairinfo[0: -4]))), 'wb')
        metric_dict = {'mix': [], 'SDR': {'ori': [], 'sep': []}, 'SIR': {'ori': [], 'sep': []}, 'SAR': {'ori': [], 'sep': []}, 'PERM': {'sep': []}}

        np.random.seed(0)
        torch.manual_seed(0)
        for idx, info in enumerate(makemix):
            sir_order = info[6]
            spkr_id_1, srcdata_path_1, offset_1, duration_1 = info[0]
            spkr_id_2, srcdata_path_2, offset_2, duration_2 = info[1]
            spkr_id_3, srcdata_path_3, offset_3, duration_3 = info[2]
            spkr_id_4, srcdata_path_4, offset_4, duration_4 = info[3]
            gidx_rir = info[4]
            # mix_sir = info[5]
            mix_sir = 0
            print("Processing {}/{} mixture.".format(idx + 1, len(makemix)))
            # Generate mixture and source signals
            src_name_1 = os.path.basename(srcdata_path_1)[0: -4]
            src_name_2 = os.path.basename(srcdata_path_2)[0: -4]
            src_name_3 = os.path.basename(srcdata_path_3)[0: -4]
            src_name_4 = os.path.basename(srcdata_path_4)[0: -4]
            out_path_necessary = os.path.join(
                out_path[path_idx], "{}-{}_{}-{}_{}-{}_{}-{}_{}".format(src_name_1, spkr_id_1, src_name_2, spkr_id_2,
                                                                        src_name_3, spkr_id_3, src_name_4, spkr_id_4, method)
            )
            if prefix is not None:
                # srcdata_path_1 = os.path.join(prefix, srcdata_path_1.split('/', 4)[-1])
                # srcdata_path_2 = os.path.join(prefix, srcdata_path_2.split('/', 4)[-1])
                srcdata_path_1 = os.path.join(prefix, 'DATASET/Librispeech/concatenate/test_clean/' + src_name_1 + '.wav')
                srcdata_path_2 = os.path.join(prefix, 'DATASET/Librispeech/concatenate/test_clean/' + src_name_2 + '.wav')
                srcdata_path_3 = os.path.join(prefix, 'DATASET/Librispeech/concatenate/test_clean/' + src_name_3 + '.wav')
                srcdata_path_4 = os.path.join(prefix, 'DATASET/Librispeech/concatenate/test_clean/' + src_name_4 + '.wav')
            src_1, _ = sf.read(srcdata_path_1, start=offset_1, stop=offset_1 + duration_1)
            src_2, _ = sf.read(srcdata_path_2, start=offset_2, stop=offset_2 + duration_2)
            src_3, _ = sf.read(srcdata_path_3, start=offset_3, stop=offset_3 + duration_3)
            src_4, _ = sf.read(srcdata_path_4, start=offset_4, stop=offset_4 + duration_4)
            # src_1 = src_1 - np.mean(src_1)
            # src_2 = src_2 - np.mean(src_2)
            # src_1 = (src_1 - np.mean(src_1)) / np.max(np.abs(src_1 - np.mean(src_1))) + np.mean(src_1)
            src_1 = src_1 / np.max(np.abs(src_1))
            src_2 = src_2 * np.std(src_1) / np.std(src_2)
            src_3 = src_3 * np.std(src_1) / np.std(src_3)
            src_4 = src_4 * np.std(src_1) / np.std(src_4)
            assert src_1.shape[0] == src_2.shape[0] == src_3.shape[0] == src_4.shape[0]

            channel = idx2rir[gidx_rir]['channel']
            rir_path_1 = os.path.join(prefix, idx2rir[gidx_rir]['path'][sir_order[0]])
            rir_path_2 = os.path.join(prefix, idx2rir[gidx_rir]['path'][sir_order[1]])
            rir_path_3 = os.path.join(prefix, idx2rir[gidx_rir]['path'][sir_order[2]])
            rir_path_4 = os.path.join(prefix, idx2rir[gidx_rir]['path'][sir_order[3]])
            rir_1 = sio.loadmat(rir_path_1)
            rir_2 = sio.loadmat(rir_path_2)
            rir_3 = sio.loadmat(rir_path_3)
            rir_4 = sio.loadmat(rir_path_4)
            rir_1 = rir_1['impulse_response'][:, channel]
            rir_2 = rir_2['impulse_response'][:, channel]
            rir_3 = rir_3['impulse_response'][:, channel]
            rir_4 = rir_4['impulse_response'][:, channel]
            rir_1 = np.stack([librosa.core.resample(rir_1[:, i], 48000, 16000) for i in range(len(channel))], axis=0)
            rir_2 = np.stack([librosa.core.resample(rir_2[:, i], 48000, 16000) for i in range(len(channel))], axis=0)
            rir_3 = np.stack([librosa.core.resample(rir_3[:, i], 48000, 16000) for i in range(len(channel))], axis=0)
            rir_4 = np.stack([librosa.core.resample(rir_4[:, i], 48000, 16000) for i in range(len(channel))], axis=0)
            rir_1 = rir_1[:, 0: int(0.8 * 16000)]
            rir_2 = rir_2[:, 0: int(0.8 * 16000)]
            rir_3 = rir_3[:, 0: int(0.8 * 16000)]
            rir_4 = rir_4[:, 0: int(0.8 * 16000)]
            # reference mic
            mix_r = [convolve(src_1, rir_1[1, :]), convolve(src_2, rir_2[1, :]), convolve(src_3, rir_3[1, :]), convolve(src_4, rir_4[1, :])]
            # scale_fac = [1 / np.std(mix_r[i]) for i in range(4)]
            # mix_r = [mix_r[i] * scale_fac[i] for i in range(4)]
            # scale_fac = scale_fac[0: 1] + [x * np.sqrt(10**(-mix_sir / 10) / (3)) for x in scale_fac[1:]]
            scale_fac = np.sqrt(np.var(mix_r[0]) * 10**(-mix_sir / 10) / sum([np.var(mix_r[i]) for i in range(1, 4)]))
            # scale_fac_2 = np.std(mix_r[0]) / np.std(mix_r[1])
            # scale_fac_3 = np.std(mix_r[0]) * 10**(-mix_sir / 20) / np.std(mix_r[2])
            # scale_fac_4 = np.std(mix_r[0]) * 10**(-mix_sir / 20) / np.std(mix_r[3])
            # mix_r = [convolve(src_1 , rir_1[1, :]), convolve(src_2 * scale_fac, rir_2[1, :]), convolve(src_3 * scale_fac, rir_3[1, :]), convolve(src_4 * scale_fac, rir_4[1, :])]
            # a = np.std(mix_r[0]) / (np.std(mix_r[1]) + np.std(mix_r[2]) + np.std(mix_r[3]))
            mix = [
                convolve(src_1, rir_1[i, :]) + convolve(src_2 * scale_fac, rir_2[i, :]) + convolve(src_3 * scale_fac, rir_3[i, :]) + convolve(src_4 * scale_fac, rir_4[i, :])
                for i in range(4)
            ]
            mix = np.stack(mix, axis=1)
            src = np.stack((src_1, src_2, src_3, src_4), axis=1)
            mix = mix[0: src.shape[0], :]
            metrics_ori = mir_eval.separation.bss_eval_sources(src.T, mix.T)
            if method.startswith("MVAE"):
                src = zero_pad(src.T, 4, hop_length=hop)
                mix = zero_pad(mix.T, 4, hop_length=hop)
                src, mix = src.T, mix.T

            # Separate mixture using method
            mix_spec = [librosa.core.stft(np.asfortranarray(mix[:, ch]), n_fft=nfft, hop_length=hop, win_length=nfft) for ch in range(mix.shape[1])]
            mix_spec = np.stack(mix_spec, axis=1)
            if method == 'ILRMA':
                sep_spec, flag = ilrma(mix_spec, 1000, n_basis=2)
            elif method == 'GMM':
                sep_spec, flag = avgmm(mix_spec, 1000, state_num=2)
            elif method == "MVAE_onehot":
                sep_spec, flag = mvae_onehot(mix_spec.swapaxes(1, 2), model, n_iter=1000, device=device)
            elif method == "MVAE_ge2e":
                sep_spec, flag = mvae_ge2e(mix_spec.swapaxes(1, 2), vae_model, spkr_model, fb_mat=fb_mat, n_iter=1000, device=device)

            sep = [librosa.core.istft(sep_spec[:, ch, :], hop_length=hop, length=src.shape[0]) for ch in range(sep_spec.shape[1])]
            sep = np.stack(sep, axis=1)
            # sep_nm = copy.deepcopy(sep)
            # mix_nm = copy.deepcopy(mix)
            # src_nm = copy.deepcopy(src)
            # sep_nm = sep_nm - np.mean(sep_nm, axis=0, keepdims=True)
            # mix_nm = mix_nm - np.mean(mix_nm, axis=0, keepdims=True)
            # src_nm = src_nm - np.mean(src_nm, axis=0, keepdims=True)
            metrics = mir_eval.separation.bss_eval_sources(src.T, sep.T)
            metrics_ori = mir_eval.separation.bss_eval_sources(src.T, mix.T)

            metric_dict['mix'].append(makemix[idx])
            for i, m in enumerate(['SDR', 'SIR', 'SAR']):
                metric_dict[m]['sep'].extend(metrics[i].tolist())
                metric_dict[m]['ori'].extend(metrics_ori[i].tolist())
            metric_dict['PERM']['sep'].append(metrics[-1])
            
            if sub_state == 0:
                sf.write(out_path_necessary + '_sep.wav', sep, fs)
                with open(os.path.join(out_path[path_idx], 'log-{}.txt'.format(os.path.basename(pairinfo[0: -4]))), 'at') as f_log:
                    print('Finish processing index {} for {}'.format(idx, os.path.basename(pairinfo[0: -4])), file=f_log)
                if flag:
                    import ipdb; ipdb.set_trace()
                    with open(os.path.join(out_path[path_idx], 'anomaly-log-{}.txt'.format(os.path.basename(pairinfo[0: -4]))), 'at') as f:
                        print(os.path.basename(out_path_necessary), file=f)
            if sub_state == 2:
                sf.write(out_path_necessary + '_mixtest.wav', mix, fs)
                sf.write(out_path_necessary + '_srctest.wav', src, fs)
                sf.write(out_path_necessary + '_septest.wav', sep, fs)

        if sub_state == 0:
            pickle.dump(metric_dict, f_metrics)
            f_metrics.close()


def get_seped_utter_3src(rirdata_path, pairinfo_path, out_path, sub_state=0, method='ILRMA', prefix=None, rep_metrics_path=None, **kwargs):
    """ separate mixtures in makemix.n.pkl and save separated utterances to out_path """
    """
    Args:
        `rirdata_path`: [str] path where idx2rir.pkl is saved
        `pairinfo_path`: [list] path where makemix.pkl is saved
        `out_path`: [list] path where .wav file will be saved
        `sub_state`: [int] 0 is execution state; 1 is debugging state
        `method`: [str] separation method "ILRMA" and "MVAE"
        `kwargs`: [dict] params for STFT and method
    Out:
        `metric_dict`: [Not returned but saved][dict]
                    dict object that saves SDR, SIR, SAR information of separated data in makemix.n.pkl order
                    {'mix': [makemix[0], makemix[1], ...], 'SDR': {'ori': [], 'sep': []}, 'SIR': {'ori': [], 'sep': []}, 'SAR': {'ori': [], 'sep': []}}
        `metrics.n.pkl`: pickle file that saves metric_dict for job n
        `log.n.txt`: txt file that saves log for job n
                     each line of log is 'Finish processing index m for job n' 
        `sep`: saved in wav file
    """
    print(sub_state)
    f_idx2rir = open(os.path.join(rirdata_path, 'idx2rir.pkl'), 'rb')
    idx2rir = pickle.load(f_idx2rir)
    f_idx2rir.close()
    nfft = int(kwargs['fs'] * kwargs['stft_len']) 
    hop = int(kwargs['fs'] * kwargs['stft_hop'])
    fs = kwargs['fs']

    if method == 'MVAE_onehot':
        modelfile_path = kwargs['vae_model_path']
        n_embedding = kwargs['embedding_dim']
        device = kwargs['device']
        model = vae.net(n_embedding)
        model.load_state_dict(torch.load(modelfile_path, map_location=device))
        model.to(device)
        model.eval()
    elif method == 'MVAE_ge2e':
        fb_mat = torch.from_numpy(librosa.filters.mel(16000, nfft, n_mels=40)).unsqueeze(0)
        vaemodel_path = kwargs['vae_model_path']
        spkrmodel_path = kwargs['spkr_model_path']
        n_embedding = kwargs['embedding_dim']
        device = kwargs['device']
        vae_model = vae.net(n_embedding)
        vae_model.load_state_dict(torch.load(vaemodel_path, map_location=device))
        vae_model.to(device)
        vae_model.eval()
        spkr_model = speaker_encoder_ge2e()
        spkr_model_dict = spkr_model.state_dict()
        pretrained_dict = torch.load(spkrmodel_path, map_location=torch.device('cpu'))
        pretrained_dict_rename = {}
        for k, v in pretrained_dict.items():
            try:
                param_name = k.split('.', 1)[1]
                pretrained_dict_rename[param_name] = v
            except IndexError:
                pass
        pretrained_dict_rename = {k: v for k, v in pretrained_dict_rename.items() if k in spkr_model_dict}
        spkr_model_dict.update(pretrained_dict_rename)
        spkr_model.load_state_dict(spkr_model_dict)
        spkr_model.cuda(device)
        spkr_model.eval()
    
    for path_idx, pairinfo in enumerate(pairinfo_path):
        print('Current makemix is {}'.format(pairinfo))
        os.makedirs(out_path[path_idx], exist_ok=True)
        
        f_makemix = open(pairinfo, 'rb')
        makemix = pickle.load(f_makemix)
        f_makemix.close()
        if sub_state == 0:
            f_metrics = open(os.path.join(out_path[path_idx], 'metrics-{}.pkl'.format(os.path.basename(pairinfo[0: -4]))), 'wb')
        metric_dict = {'mix': [], 'SDR': {'ori': [], 'sep': []}, 'SIR': {'ori': [], 'sep': []}, 'SAR': {'ori': [], 'sep': []}, 'PERM': {'sep': []}}

        np.random.seed(0)
        torch.manual_seed(0)
        idx_list = [8, 12, 18, 27, 39, 41, 48, 74, 86, 99]
        makemix = [makemix[i] for i in idx_list]
        for idx, info in enumerate(makemix):
            # print(idx)
            # sir_order = [0, 1, 2]
            sir_order = info[5]
            spkr_id_1, srcdata_path_1, offset_1, duration_1 = info[0]
            spkr_id_2, srcdata_path_2, offset_2, duration_2 = info[1]
            spkr_id_3, srcdata_path_3, offset_3, duration_3 = info[2]
            gidx_rir = info[3]
            mix_sir = info[4]
            # mix_sir = -5

            print("Processing {}/{} mixture.".format(idx + 1, len(makemix)))
            # Generate mixture and source signals
            src_name_1 = os.path.basename(srcdata_path_1)[0: -4]
            src_name_2 = os.path.basename(srcdata_path_2)[0: -4]
            src_name_3 = os.path.basename(srcdata_path_3)[0: -4]
            out_path_necessary = os.path.join(
                out_path[path_idx], "{}-{}_{}-{}_{}-{}_{}".format(src_name_1, spkr_id_1, src_name_2, spkr_id_2,
                                                                  src_name_3, spkr_id_3, method)
            )
            # out_path_necessary = os.path.join(
            #     out_path[path_idx], "{}-{}_{}-{}_{}-{}".format(src_name_1, spkr_id_1, src_name_2, spkr_id_2,
            #                                                       src_name_3, spkr_id_3)
            # )
            if prefix is not None:
                # srcdata_path_1 = os.path.join(prefix, srcdata_path_1.split('/', 4)[-1])
                # srcdata_path_2 = os.path.join(prefix, srcdata_path_2.split('/', 4)[-1])
                srcdata_path_1 = os.path.join(prefix, 'DATASET/Librispeech/concatenate/test_clean/' + src_name_1 + '.wav')
                srcdata_path_2 = os.path.join(prefix, 'DATASET/Librispeech/concatenate/test_clean/' + src_name_2 + '.wav')
                srcdata_path_3 = os.path.join(prefix, 'DATASET/Librispeech/concatenate/test_clean/' + src_name_3 + '.wav')
            src_1, _ = sf.read(srcdata_path_1, start=offset_1, stop=offset_1 + duration_1)
            src_2, _ = sf.read(srcdata_path_2, start=offset_2, stop=offset_2 + duration_2)
            src_3, _ = sf.read(srcdata_path_3, start=offset_3, stop=offset_3 + duration_3)
            # src_1 = src_1 - np.mean(src_1)
            # src_2 = src_2 - np.mean(src_2)
            # src_1 = (src_1 - np.mean(src_1)) / np.max(np.abs(src_1 - np.mean(src_1))) + np.mean(src_1)
            src_1 = src_1 / np.max(np.abs(src_1))
            src_2 = src_2 * np.std(src_1) / np.std(src_2)
            src_3 = src_3 * np.std(src_1) / np.std(src_3)
            assert src_1.shape[0] == src_2.shape[0] == src_3.shape[0]

            channel = idx2rir[gidx_rir]['channel']
            rir_path_1 = os.path.join(prefix, idx2rir[gidx_rir]['path'][sir_order[0]])
            rir_path_2 = os.path.join(prefix, idx2rir[gidx_rir]['path'][sir_order[1]])
            rir_path_3 = os.path.join(prefix, idx2rir[gidx_rir]['path'][sir_order[2]])
            rir_1 = sio.loadmat(rir_path_1)
            rir_2 = sio.loadmat(rir_path_2)
            rir_3 = sio.loadmat(rir_path_3)
            rir_1 = rir_1['impulse_response'][:, channel]
            rir_2 = rir_2['impulse_response'][:, channel]
            rir_3 = rir_3['impulse_response'][:, channel]
            rir_1 = np.stack([librosa.core.resample(rir_1[:, i], 48000, 16000) for i in range(len(channel))], axis=0)
            rir_2 = np.stack([librosa.core.resample(rir_2[:, i], 48000, 16000) for i in range(len(channel))], axis=0)
            rir_3 = np.stack([librosa.core.resample(rir_3[:, i], 48000, 16000) for i in range(len(channel))], axis=0)
            rir_1 = rir_1[:, 0: int(0.8 * 16000)]
            rir_2 = rir_2[:, 0: int(0.8 * 16000)]
            rir_3 = rir_3[:, 0: int(0.8 * 16000)]
            # reference mic
            mix_r = [convolve(src_1, rir_1[1, :]), convolve(src_2, rir_2[1, :]), convolve(src_3, rir_3[1, :])]

            scale_fac = np.sqrt(np.var(mix_r[0]) * 10**(-mix_sir / 10) / sum([np.var(mix_r[i]) for i in range(1, 3)]))
            # scale_fac_2 = np.std(mix_r[0]) / np.std(mix_r[1])
            # scale_fac_3 = np.std(mix_r[0]) * 10**(-mix_sir / 20) / np.std(mix_r[2])
            mix = [
                convolve(src_1, rir_1[i, :]) + convolve(src_2 * scale_fac, rir_2[i, :]) + convolve(src_3 * scale_fac, rir_3[i, :])
                for i in range(3)
            ]
            mix = np.stack(mix, axis=1)
            src = np.stack((src_1, src_2, src_3), axis=1)
            mix = mix[0: src.shape[0], :]
            metrics_ori = mir_eval.separation.bss_eval_sources(src.T, mix.T)
            if method.startswith("MVAE"):
                src = zero_pad(src.T, 4, hop_length=hop)
                mix = zero_pad(mix.T, 4, hop_length=hop)
                src, mix = src.T, mix.T

            # Separate mixture using method
            mix_spec = [librosa.core.stft(np.asfortranarray(mix[:, ch]), n_fft=nfft, hop_length=hop, win_length=nfft) for ch in range(mix.shape[1])]
            mix_spec = np.stack(mix_spec, axis=1)
            if method == 'ILRMA':
                sep_spec, flag = myilrma(mix_spec, 1000, n_basis=2)
                # sep_spec = ilrma(mix_spec.transpose(2, 0, 1), n_iter=200, n_components=2)
                # sep_spec = sep_spec.transpose(1, 2, 0)
            elif method == "chainlike":
                sep_spec, flag = chainlike(mix_spec, n_iter=1000, clique_bins=128, clique_hop=1)
            # elif method == "chainlike_prob":
            #     sep_spec, flag = chainlike_prob(mix_spec, n_iter=1000, clique_bins=256, clique_hop=128)
            elif method == 'GMM':
                sep_spec, flag = avgmm(mix_spec, 1000, state_num=2)
            elif method == "MVAE_onehot":
                sep_spec, flag = mvae_onehot(mix_spec.swapaxes(1, 2), model, n_iter=1000, device=device)
            elif method == "MVAE_ge2e":
                sep_spec, flag = mvae_ge2e(mix_spec.swapaxes(1, 2), vae_model, spkr_model, fb_mat=fb_mat, n_iter=1000, device=device)

            sep = [librosa.core.istft(sep_spec[:, ch, :], hop_length=hop, length=src.shape[0]) for ch in range(sep_spec.shape[1])]
            sep = np.stack(sep, axis=1)
            # sep_nm = copy.deepcopy(sep)
            # mix_nm = copy.deepcopy(mix)
            # src_nm = copy.deepcopy(src)
            # sep_nm = sep_nm - np.mean(sep_nm, axis=0, keepdims=True)
            # mix_nm = mix_nm - np.mean(mix_nm, axis=0, keepdims=True)
            # src_nm = src_nm - np.mean(src_nm, axis=0, keepdims=True)
            metrics = mir_eval.separation.bss_eval_sources(src.T, sep.T)
            metrics_ori = mir_eval.separation.bss_eval_sources(src.T, mix.T)

            metric_dict['mix'].append(makemix[idx])
            for i, m in enumerate(['SDR', 'SIR', 'SAR']):
                metric_dict[m]['sep'].extend(metrics[i].tolist())
                metric_dict[m]['ori'].extend(metrics_ori[i].tolist())
            metric_dict['PERM']['sep'].append(metrics[-1])
            
            if sub_state == 0:
                sf.write(out_path_necessary + '_sep.wav', sep, fs)
                # sf.write(out_path_necessary + '_srctest.wav', src, fs)
                with open(os.path.join(out_path[path_idx], 'log-{}.txt'.format(os.path.basename(pairinfo[0: -4]))), 'at') as f_log:
                    print('Finish processing index {} for {}'.format(idx, os.path.basename(pairinfo[0: -4])), file=f_log)
                if flag:
                    import ipdb; ipdb.set_trace()
                    with open(os.path.join(out_path[path_idx], 'anomaly-log-{}.txt'.format(os.path.basename(pairinfo[0: -4]))), 'at') as f:
                        print(os.path.basename(out_path_necessary), file=f)
            if sub_state == 1:
                sf.write(out_path_necessary + '_mixtest.wav', mix, fs)
                sf.write(out_path_necessary + '_srctest.wav', src, fs)
                sf.write(out_path_necessary + '_septest.wav', sep, fs)

        if sub_state == 0:
            pickle.dump(metric_dict, f_metrics)
            f_metrics.close()

def get_rkg_ret_3src(datafile_list, device, sep_method, spkr_encoder="ge2e_withaug", rkg_method="cossim",
                     enroll_len=30, fs=16000, spkr_model=None, use_length_norm=True, sub_state=0,
                     xvecmodel_path=None, ge2emodel_path=None, xvec_path=None, lda_path=None, plda_path=None,
                     **kwargs):
    """ 
    Args:
        `datafile_list`: [str] path where makemix_xxx.pkl is saved
        `sep_method`: [str] name of separation method

    Out:
        `ret`: [Not returned but saved] 
               {"0.15-20": {"object": [file_idx-1-0, file_idx-1-1, file_idx-2-0, ...], "rkg": [rkg-for-file_idx-1-0, rkg-for-file_idx-1-1, ...]}}
    """
    ret = {'object': [], 'rkg': []}
    print(sep_method)
    print(spkr_encoder)
    print(rkg_method)
    if rkg_method.startswith("plda"):
        print("speakermodel_path: {}".format(xvecmodel_path))
        print("pldatraindata_path: {}".format(xvec_path))
        print("lda_path: {}".format(lda_path))
        print("plda_path: {}".format(plda_path))
        print("use_length_normalization: {}".format(use_length_norm))
    else:
        print("speakermodel_path: {}".format(ge2emodel_path))


    # Prepare 
    if spkr_encoder.startswith('ge2e'):
        fb_mat = torch.from_numpy(librosa.filters.mel(hp.data.sr, n_fft, n_mels=hp.model.nmels)).unsqueeze(0).cuda(device)
    elif spkr_encoder.startswith('xvec'):
        melsetting = {}
        melsetting['n_fft'] = n_fft
        melsetting['win_length'] = n_fft
        melsetting['hop_length'] = hop
        melsetting['n_mels'] = hp.model.feat_num
        transform = torchaudio.transforms.MFCC(sample_rate=hp.data.sr, n_mfcc=hp.model.feat_num, melkwargs=melsetting)
    # Load speaker model
    if spkr_encoder.startswith('ge2e'):
        spkr_model = speaker_encoder_ge2e()
        spkr_model_dict = spkr_model.state_dict()
        pretrained_dict = torch.load(os.path.join(hp.path.prefix, ge2emodel_path), map_location=torch.device('cpu'))
        pretrained_dict_rename = {}
        for k, v in pretrained_dict.items():
            try:
                param_name = k.split('.', 1)[1]
                pretrained_dict_rename[param_name] = v
            except IndexError:
                pass
    elif spkr_encoder.startswith('xvec'):
        spkr_model = speaker_encoder_xvec()
        spkr_model_dict = spkr_model.state_dict()
        pretrained_dict = torch.load(os.path.join(hp.path.prefix, xvecmodel_path), map_location=torch.device('cpu'))
        pretrained_dict_rename = pretrained_dict
    pretrained_dict_rename = {k: v for k, v in pretrained_dict_rename.items() if k in spkr_model_dict}
    spkr_model_dict.update(pretrained_dict_rename)
    spkr_model.load_state_dict(spkr_model_dict)
    spkr_model.cuda(device)
    spkr_model.eval()

    # recognize one by one
    for datafile in datafile_list:
        print(datafile)
        with open(datafile, 'rb') as f:
            makemix = pickle.load(f)
        dataname = os.path.basename(datafile)[0: -4]   # e.g. makemix_same_00
        ret = {"object": [], "rkg": []}
        target_path = os.path.join(os.path.dirname(datafile), sep_method, "128-1")  # path where separated .wav is saved  #  
        if sub_state == 0:
            with open(os.path.join(target_path, 'rkg_ret-{}-{}-{}.pkl.info.txt').format(dataname, spkr_encoder, rkg_method), 'w') as f:
                print("speakermodel_path: {}".format(xvecmodel_path if spkr_encoder.startswith('xvec') else ge2emodel_path), file=f)
                if rkg_method.startswith("plda"):
                    print("pldatraindata_path: {}".format(xvec_path), file=f)
                    print("lda_path: {}".format(lda_path), file=f)
                    print("plda_path: {}".format(plda_path), file=f)
                    print("use_length_normalization: {}".format(use_length_norm), file=f)
            f_txt = open(os.path.join(target_path, 'rkg_log-{}-{}-{}.txt'.format(dataname, spkr_encoder, rkg_method)), 'wt')

        for file_idx, info in enumerate(makemix):
            spkr_info = info[0:3]
            spkr_info = list(zip(*spkr_info))
            # print(file_idx)
            ret['object'].append("{}".format(file_idx))
            src_name = [os.path.basename(path)[0: -4] for path in spkr_info[1]]
            sep_path = os.path.join(
                target_path, "{}-{}_{}-{}_{}-{}_{}_sep.wav".format(src_name[0], spkr_info[0][0], src_name[1], spkr_info[0][1], #"MVAE_onehot" if sep_method == "MVAE_onehot--ilrmainit" else sep_method
                                                                   src_name[2], spkr_info[0][2], sep_method))
            srcdata_path = [os.path.join(hp.path.prefix, 'DATASET/Librispeech/concatenate/test_clean/' + name + '.wav') for name in src_name]
            src = [sf.read(srcdata_path[i], start=spkr_info[2][i], stop=spkr_info[2][i] + spkr_info[3][i]) for i in range(len(src_name))]
            src, _ = list(zip(*src))
            enroll, _ = sf.read(srcdata_path[0], start=0, stop=int(enroll_len * fs))
            src = np.stack(src, axis=1)
            if sep_method.startswith("MVAE"):
                src = zero_pad(src.T, 4, hop_length=hop)
                src = src.T
            enroll = np.expand_dims(enroll, axis=1)
            sep, _ = sf.read(sep_path)
            sep_nm = copy.deepcopy(sep)
            src_nm = copy.deepcopy(src)
            _, _, _, perm = mir_eval.separation.bss_eval_sources(src_nm.T, sep_nm.T)

            sep = torch.from_numpy(sep.T)
            enroll = torch.from_numpy(enroll.T)
            with torch.no_grad():
                if spkr_encoder.startswith('xvec'):
                    sep = transform(sep.float())
                    sep = sep.float().cuda(device)
                    sep = (sep - sep.mean(dim=-1, keepdim=True)) / (sep.std(dim=-1, keepdim=True))
                    enroll = transform(enroll.float())
                    enroll = enroll.float().cuda(device)
                    enroll = (enroll - enroll.mean(dim=-1, keepdim=True)) / (enroll.std(dim=-1, keepdim=True))
                elif spkr_encoder.startswith('ge2e'):
                    sep = my_spectrogram(sep.float().cuda(device), n_fft, hop)
                    sep = spectrogram_normalize(sep)
                    sep = torch.matmul(fb_mat, sep)
                    sep = 10 * torch.log10(torch.clamp(sep, 1e-10))
                    enroll = my_spectrogram(enroll.float().cuda(device), n_fft, hop)
                    enroll = spectrogram_normalize(enroll)
                    enroll = torch.matmul(fb_mat, enroll)
                    enroll = 10 * torch.log10(torch.clamp(enroll, 1e-10))
                vec_sep = spkr_model.extract_embd(sep)
                vec_enroll = spkr_model.extract_embd(enroll)

                if rkg_method == "cossim":
                    sim_enroll = F.cosine_similarity(vec_sep, vec_enroll, dim=1)
                    target_idx_enroll = sim_enroll.argmax()

                elif rkg_method.startswith("plda"):
                    vec_sep = vec_sep.cpu().numpy()
                    vec_enroll = vec_enroll.cpu().numpy()
                    embd_mean = kwargs["embedding_mean"]
                    lda_machine = kwargs["lda_machine"]
                    plda_base = kwargs["plda_base"]
                    vec_sep = lda_machine.forward(vec_sep - embd_mean)
                    vec_enroll = lda_machine.forward(vec_enroll - embd_mean)
                    if use_length_norm:
                        # vec_sep = vec_sep @ whitening_matrix.T
                        # vec_enroll = vec_enroll @ whitening_matrix.T
                        vec_sep = vec_sep / np.linalg.norm(vec_sep, axis=1, keepdims=True)
                        vec_enroll = vec_enroll / np.linalg.norm(vec_sep, axis=1, keepdims=True)
                    vec_sep = vec_sep.astype(np.float64)
                    vec_enroll = vec_enroll.astype(np.float64)
                    plda_machine = bob.learn.em.PLDAMachine(plda_base)
                    plda_trainer = bob.learn.em.PLDATrainer()
                    plda_trainer.enroll(plda_machine, vec_enroll)
                    loglike_enroll = np.stack([plda_machine.compute_log_likelihood(vec_sep[0, :]), plda_machine.compute_log_likelihood(vec_sep[1, :]), 
                                               plda_machine.compute_log_likelihood(vec_sep[2, :])])
                    target_idx_enroll = loglike_enroll.argmax()
                    # loglike_enroll_1 = np.stack([
                    #     plda_machine.compute_log_likelihood(np.stack((vec_enroll[0, :], vec_sep[0, :]))), 
                    #     plda_machine.compute_log_likelihood(np.stack((vec_enroll[0, :], vec_sep[1, :]))),
                    #     plda_machine.compute_log_likelihood(np.stack((vec_enroll[0, :], vec_sep[2, :])))])

                ret['rkg'].append(True if target_idx_enroll == perm[0] else False)
                if sub_state == 0:
                    print("{}-{}_{}-{}_{}-{}_{}_sep.wav: {}".format(src_name[0], spkr_info[0][0], src_name[1], spkr_info[0][1],
                                                                    src_name[2], spkr_info[0][2], sep_method, target_idx_enroll == perm[0]), file=f_txt)
        if sub_state == 0:
            with open(os.path.join(target_path, 'rkg_ret-{}-{}-{}.pkl').format(dataname, spkr_encoder, rkg_method), 'wb') as f:
                pickle.dump(ret, f)
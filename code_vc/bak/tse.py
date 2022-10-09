class TSE(Inference):
    def __init__(self, config, suffix=''):
        super(TSE, self).__init__(config)
        self.config.path.abs_out_dir = os.path.join(self.config.path.common_prefix, self.config.path.out_dir, suffix)
        self.d_embedding = self.config.NetInput.n_embedding
        os.makedirs(self.config.path.abs_out_dir, exist_ok=True)
        os.system(f"cp {os.path.join(os.path.dirname(__file__), 'config.py')} {os.path.join(self.config.path.abs_out_dir, 'Config.py')}")

        self.ret = {'MixturePath': [], 'SepBaseName': [], 'SDROri': [], 'SDR': [], 'SIROri': [], 'SIR': [], 'SAROri': [], 'SAR': [], 'Alignment': []}

    def parse_data(self):
        excel_reader = pd.ExcelFile(os.path.join(self.config.path.common_prefix, self.config.path.data))
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
                rir = sio.loadmat(os.path.join(self.config.path.common_prefix, rir_paths[rir_id]))
                rir = rir['impulse_response'][:, channel].T
                rir = librosa.core.resample(np.asfortranarray(rir), 48000, self.fs)
                rir = rir[:, : int(0.8 * self.fs)]
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

    def target_speech_extraction(self, num_tests=None, rir_path=None, rir_mode=None, bss_method=None,
                                 target_seglen=200, target_seghop=0.5, enroll_len=30, enroll_mode='all', enrollutt_mode='enroll',
                                 mvae_config=None):
        if num_tests is None:
            num_tests = self.df[0].shape[0]
        if self.df[0].shape[0] < num_tests:
            raise ValueError(f"Number of tests {num_tests} exceeds the number of test data {self.df[0].shape[0]}!")
        with open(os.path.join(self.config.path.common_prefix, rir_path), 'rb') as f_idx2rir:
            self.idx2rir = pickle.load(f_idx2rir)
        self.rir_mode = rir_mode
        self.enroll_len = int(self.fs * enroll_len)
        if bss_method == 'MVAE':
            os.system(f"cp {os.path.join(os.path.dirname(__file__), 'BSSAlgorithm/mvae.py')} {os.path.join(self.config.path.abs_out_dir, 'mvae.py')}")

        for test_id in range(num_tests):
            # rir and sir
            self.rir_idx = self.rir_list[test_id]
            self.mix_sir = self.sir_list[test_id] if self.sir_list is not None else 0.0

            self.order_list = list(range(self.src_num))
            self.mix_src_basename = [os.path.basename(list(self.df[i].loc[test_id])[0]).split('-')[1].rsplit('.', 1)[0] for i in self.order_list]
            self.mix_src_basename = f'spkr-{self.mix_src_basename[0]}_spkr-' + '-'.join(self.mix_src_basename[1:])
            self.src_path = os.path.join(os.path.dirname(self.config.path.abs_out_dir), f"{self.mix_src_basename}_src.wav")
            self.mix_path = os.path.join(os.path.dirname(self.config.path.abs_out_dir), f"{self.mix_src_basename}_mix.wav")

            # read source signals
            if os.path.exists(self.src_path):
                self.src_sigs, _ = sf.read(self.src_path)
                self.src_sigs = self.src_sigs.T
            else:
                self.src_sigs = []
                for idx, src_id in enumerate(self.order_list):
                    utt_path, utt_offset, utt_dur, _ = self.df[src_id].loc[test_id]
                    src_sig, raw_sr = sf.read(os.path.join(self.config.path.data_prefix, utt_path), start=utt_offset, stop=utt_offset + utt_dur)
                    if raw_sr != self.fs:
                        src_sig = librosa.core.resample(src_sig, raw_sr, self.fs)
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

            for target_id in range(self.src_num):
                self.order_list_new = [self.order_list[target_id]] + self.order_list[: target_id] + self.order_list[target_id + 1:]
                self.sep_basename = [os.path.basename(list(self.df[i].loc[test_id])[0]).split('-')[1].rsplit('.', 1)[0] for i in self.order_list_new]
                self.sep_basename = f'target-{self.sep_basename[0]}_inferences-' + '-'.join(self.sep_basename[1:])
                self.sep_path = os.path.join(self.config.path.abs_out_dir, f"{self.sep_basename}_sep.wav")

                # read enrollment signals
                if enroll_mode == 'None':
                    spkr_embd = torch.ones(0, self.d_embedding).to(self.config.general.device)
                else:
                    self.enroll_sigs = []
                    enroll_num = 1 if enroll_mode == 'target' else self.src_num
                    for idx in range(enroll_num):
                        if enrollutt_mode == 'enroll':
                            enroll_path, _, _, _ = self.df[self.order_list_new[idx]].loc[test_id]
                            enroll_sig, raw_sr = sf.read(os.path.join(self.config.path.data_prefix, enroll_path), start=0, stop=self.enroll_len)
                            if raw_sr != self.fs:
                                enroll_sig = librosa.core.resample(enroll_sig, raw_sr, self.fs)
                            enroll_sig = enroll_sig - np.mean(enroll_sig)
                            enroll_sig = enroll_sig / np.max(np.abs(enroll_sig))
                            self.enroll_sigs.append(enroll_sig)
                        elif enrollutt_mode == 'source':
                            self.enroll_sigs.append(self.src_sigs[self.order_list_new[idx], :])
                    enroll_sig = torch.from_numpy(np.stack(self.enroll_sigs, axis=0)).to(self.config.general.device).float()
                    enroll_feat, _, _ = self.transform(enroll_sig)
                    # generate spkr_embd
                    with torch.no_grad():
                        spkr_embd = self.model.get_speaker_embeddings(enroll_feat, target_seglen=target_seglen, target_seghop=target_seghop)

                # BSS
                self.mix_spec = np.stack([
                    librosa.core.stft(np.asfortranarray(self.mix_sigs[ch, :]), n_fft=self.stft_len, hop_length=self.hop_len)
                    for ch in range(self.mix_sigs.shape[0])
                ], axis=1)
                if bss_method == 'ILRMA':
                    sep_spec, flag = myilrma(self.mix_spec, 1000, n_basis=2)
                elif bss_method == 'MVAE':
                    sep_spec, flag = MVAEiva(self.mix_spec, spkr_embd, self.model,
                                             device=self.config.general.device, **mvae_config)
                self.sep_sigs = np.stack([
                    librosa.core.istft(sep_spec[:, ch, :], hop_length=self.hop_len, length=self.src_sigs.shape[1])
                    for ch in range(sep_spec.shape[1])
                ], axis=0)
                sf.write(self.sep_path, self.sep_sigs.T, self.fs)

                # metric
                met_before = mir_eval.separation.bss_eval_sources(self.src_sigs, np.stack([self.mix_sigs[0, :] for _ in range(self.src_num)], axis=0))
                met_target = mir_eval.separation.bss_eval_sources(self.src_sigs, np.stack([self.sep_sigs[0, :] for _ in range(self.src_num)], axis=0))
                met_after = mir_eval.separation.bss_eval_sources(self.src_sigs, self.sep_sigs)
                self.ret['MixturePath'].append(os.path.join(os.path.dirname(self.config.path.out_dir), f"{self.mix_src_basename}_mix.wav"))
                self.ret['SepBaseName'].append(f"{self.sep_basename}_sep.wav")
                print(met_target[1][target_id])
                for i, m in enumerate(['SDR', 'SIR', 'SAR']):
                    if enroll_mode == 'None' or bss_method != 'MVAE' or (bss_method == "MVAE" and mvae_config['ilrma_init'] is True):
                        self.ret[m].append(met_after[i][target_id])
                    else:
                        self.ret[m].append(met_target[i][target_id])
                    self.ret[m + 'Ori'].append(met_before[i][target_id])
                self.ret['Alignment'].append(True if met_after[-1][target_id] == 0 else False)
        self.ret_path = os.path.join(self.config.path.common_prefix, self.config.path.abs_out_dir, 'Result.xls')
        df = pd.DataFrame(self.ret)
        excel_writer = pd.ExcelWriter(self.ret_path)
        df.to_excel(excel_writer, index_label="Index")
        excel_writer.save()
        excel_writer.close()

    def __call__(self, config, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.target_speech_extraction(**config)


    def CheckData(self):
        random.seed(seed)
        torch.random.manual_seed(seed)
        transform = Spectrogram(**cfgs.sigproc)

        train_dataset = CleanDataset.from_pkl(cfgs.path.train, cfgs.general.data_prefix)
        vad_dataset = CleanDataset.from_pkl(cfgs.path.vad, cfgs.general.data_prefix)
        diagnosis_dataset = CleanDataset.from_pkl(cfgs.path.diagnosis, cfgs.general.data_prefix)

        train_sampler = TrainBatchSampler.from_dataset(train_dataset, cfgs.train.batch_size, n_batch=None, drop_last=False)
        vad_sampler = TrainBatchSampler.from_dataset(vad_dataset, cfgs.train.batch_size, n_batch=cfgs.train.num_batch_vad, drop_last=True)
        diagnosis_sampler = TrainBatchSampler.from_dataset(diagnosis_dataset, cfgs.train.batch_size, n_batch=cfgs.train.num_batch_vad, drop_last=True)

        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=CollateFnWrapper, pin_memory=True)
        vad_loader = DataLoader(vad_dataset, batch_sampler=vad_sampler, num_workers=0, collate_fn=CollateFnWrapper, pin_memory=True)
        diagnosis_loader = DataLoader(diagnosis_dataset, batch_sampler=diagnosis_sampler, num_workers=0, collate_fn=CollateFnWrapper, pin_memory=True)

        mesg = [
            'Number of speakers: {}'.format(train_dataset.spkr_num),
            "Train_loader length: {}".format(int(len(train_loader))),
            'Vad_loader length: {}'.format(int(len(vad_loader))),
            'Diagnosis_loader length: {}'.format(int(len(diagnosis_loader))),
            '\n',
        ]
        for mes in mesg:
            print(mes)
            if not cfgs.train.resume:
                logger.info(mes)

        data_mean_all = []
        data_std_all = []
        for in_data in train_loader:
            data = in_data.data
            uttinfo = in_data.uttinfo
            data = transform(data.float())
            data = data.float().cuda(cfgs.general.device)
            for idx, samp in enumerate(data):
                data_mean = torch.mean(samp.mean(dim=-1)).item()
                data_std = torch.mean(samp.std(dim=-1)).item()
                if data_std < 1:
                    print("std smaller than 1")
                    import ipdb; ipdb.set_trace()
                messages = "data_mean: {:.4f}, data_std: {:.4f}, uttinfo: {}".format(data_mean, data_std, uttinfo[idx])
                print(messages)
                logger.info(messages)
                data_mean_all.append(abs(data_mean))
                data_std_all.append(data_std)
        messages = [
            "AVG_mean: {:.4f}, AVG_std: {:.4f}".format(stat.mean(data_mean_all), stat.mean(data_std_all)),
            "mean_min: {:.4f}, mean_max: {:.4f}, std_min: {:.4f}, std_max: {:.4f}".format(min(data_mean_all), max(data_mean_all), min(data_std_all), max(data_std_all)),
            "mean_min_idx: {}, mean_max_idx: {}, std_min_idx: {}, std_max_idx: {}".format(data_mean_all.index(min(data_mean_all)), data_mean_all.index(max(data_mean_all)),
                                                                                        data_std_all.index(min(data_std_all)), data_std_all.index(max(data_std_all)))
        ]
        for mess in messages:
            print(mess)
            logger.info(mess)
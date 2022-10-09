import os
import time
import torch
import random
import IPython
import logging
import numpy as np
import statistics as stat
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import TrainBatchSampler
from network_cnn import AutoEncoder, Loss
from PreProcess.interact import interact
from PreProcess.data_utils import Spectrogram
from PreProcess.utils import cal_drop_p, parse_dropout_strategy, PlotSpectrogram


class SimpleCollate():
    def __init__(self, batch):
        data = batch
        self.data = torch.stack(data, axis=0)

        # remark, data_target, data_noise, mixinfo = list(zip(*batch))
        # self.remark = remark
        # self.data_target = torch.stack(data_target, axis=0)
        # self.data_noise = torch.stack(data_noise, axis=0)
        # self.mixinfo = mixinfo

    def pin_memory(self):
        self.data = self.data.pin_memory()

        # self.data_target = self.data_target.pin_memory()
        # self.data_noise = self.data_noise.pin_memory()
        return self


def CollateFnWrapper(batch):
    return SimpleCollate(batch)


class Transform(object):

    def __init__(self, sigproc_param):
        super(Transform, self).__init__()
        self.trans = Spectrogram(**sigproc_param)

    def __call__(self, remark, data_target, data_noise, mix_info):
        out = []
        target_spec, _, _ = self.trans(data_target)
        noise_spec, _, _ = self.trans(data_noise)
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
                # ori_energy_target = torch.mean(target_spec[idx, :, :], dim=1, keepdim=True)
                # ori_energy_noise = torch.mean(noise_spec[idx, :, :], dim=1, keepdim=True)
                # noise_spec[idx, :, :] = this_mix_info * (ori_energy_target / ori_energy_noise) * noise_spec[idx, :, :] 
                # noisy_spec = noise_spec[idx, :, :] + target_spec[idx, :, :]
                # noisy_spec = noisy_spec * ori_energy_target / torch.mean(noisy_spec, dim=1, keepdim=True)
                # print(torch.mean(this_mix_info))
                noisy_spec = (1 - this_mix_info) * target_spec[idx, :, :] + this_mix_info * noise_spec[idx, :, :]
                out.append(noisy_spec)
        out = torch.stack(out, dim=0)
        return target_spec.float(), out.float()


class Training():

    def __init__(self, cfgs, seed):
        self.cfgs = cfgs
        self.seed = seed
        self.cmd_dict = {'I': IPython.embed}  # 'H': lambda: self.change_lr(self.learning_rate / 2), #  'S': self._interrupted, #  'R': lambda: self.load(self.save_list[-1])}
        self.interact = interact(self.cmd_dict)
        self.interact.start()

        if cfgs.info.mode == 'train':
            torch.cuda.set_device(cfgs.general.device)
            if cfgs.train.resume is False:
                # backup network.py and conf.py
                os.system("cp {} {}".format(os.path.join(os.path.dirname(__file__), "network.py"), os.path.join(cfgs.path.root, 'network.py')))
                os.system("cp {} {}".format(os.path.join(os.path.dirname(__file__), "config.py"), os.path.join(cfgs.path.model, 'config.txt')))
                # setting logger
                hd = logging.FileHandler(os.path.join(cfgs.path.model, 'train.log'), mode='w')
            else:
                hd = logging.FileHandler(os.path.join(cfgs.path.model, 'train.log'), mode='a')
            formatter = logging.Formatter(fmt="{asctime}: {message}", datefmt="%Y-%m-%d %H:%M:%S", style='{')
            hd.setLevel(logging.INFO)
            hd.setFormatter(formatter)
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(hd)
            for mes in cfgs.pre_log:
                print(mes)
                self.logger.info(mes)
            self.TrainNewModel()
        elif cfgs.info.mode == 'checkdata':
            """ Data checking """
            assert cfgs.general.train is False
            hd = logging.FileHandler(os.path.join(cfgs.path.model, 'checkdata.log'), mode='w')
            formatter = logging.Formatter(fmt="{asctime}: {message}", datefmt="%Y-%m-%d %H:%M:%S", style='{')
            hd.setLevel(logging.INFO)
            hd.setFormatter(formatter)
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(hd)
            self.CheckData()

    def checknoisy(data):
        _ = PlotSpectrogram(data**0.5, 16000, 256)

    def Validate(self, dataloader, database='Dev'):
        self.model.eval()
        total_loss = 0.0
        total_isdiv = 0.0
        with torch.no_grad():
            for i, in_data in enumerate(dataloader):
                if self.cfgs.info.stage == 2:
                    data_target = in_data.data_target.to(self.cfgs.general.device)
                    data_noise = in_data.data_noise.to(self.cfgs.general.device)
                    mix_info = in_data.mixinfo
                    remark = in_data.remark
                    data_target, data_noisy = self.transform(remark, data_target, data_noise, mix_info)
                    spkr_embd, content_mu, content_log_var, x_log_var = self.model(data_target, data_noisy)
                elif self.cfgs.info.stage == 1:
                    data_target = in_data.data.to(self.cfgs.general.device)
                    data_target, _, _ = self.transform(data_target)
                    spkr_embd, content_mu, content_log_var, x_log_var = self.model(data_target.float())
                loss, is_div = self.loss_class(data_target, content_mu, content_log_var, x_log_var)
                total_loss += loss.item()
                total_isdiv += is_div.item()
                if (i + 1) % self.cfgs.train.logging_interval == 0 or (i + 1) == len(dataloader):
                    mesg = database + "Step[{}/{}], Loss:{:.4e}, ISDiv: {:.4e}".format(i + 1, len(dataloader), total_loss / (i + 1), total_isdiv / (i + 1))
                    print(time.ctime() + ': ' + mesg)
                    self.logger.info(mesg)
            return total_loss / (i + 1), total_isdiv / (i + 1)

    def TrainOneIteration(self, in_data):
        nan_flag = False
        self.model.train()
        if self.cfgs.info.stage == 2:
            data_target = in_data.data_target.to(self.cfgs.general.device)
            data_noise = in_data.data_noise.to(self.cfgs.general.device)
            mix_info = in_data.mixinfo
            remark = in_data.remark
            with torch.no_grad():
                data_target, data_noisy = self.transform(remark, data_target, data_noise, mix_info)
            # for i in range(64):
            #     print(remark[i])
            #     checknoisy(data_noisy[i, :, :])
            #     import ipdb; ipdb.set_trace()
            spkr_embd, content_mu, content_log_var, x_log_var = self.model(data_target, data_noisy)
        elif self.cfgs.info.stage == 1:
            data_target = in_data.data.to(self.cfgs.general.device)
            with torch.no_grad():
                data_target, _, _ = self.transform(data_target)
            spkr_embd, content_mu, content_log_var, x_log_var = self.model(data_target.float())
        loss, is_div = self.loss_class(data_target, content_mu, content_log_var, x_log_var)
        """ Check for NAN """
        if torch.isnan(loss).any():
            nan_flag = True
            if self.cfgs.train.pause_when_nan:
                import ipdb; ipdb.set_trace()
        """ Back propagation """
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfgs.opt_strategy.use_grad_clip:
            grad_norm = eval(self.cfgs.opt_strategy.grad_clip_strategy)
        else:
            grad_norm = -1.
        self.optimizer.step()
        return loss.item(), is_div.item(), grad_norm, nan_flag

    def Train(self):
        if self.resume_path is None and os.path.exists(os.path.join(self.cfgs.path.model, self.tensorboard_title)):
            os.system(f"rm -r {os.path.join(self.cfgs.path.model, self.tensorboard_title)}")
        self.writer = SummaryWriter(os.path.join(self.cfgs.path.model, self.tensorboard_title))
        if self.resume_path is not None:
            """ Training from checkpoint """
            checkpoint = torch.load(self.resume_path, map_location=self.cfgs.general.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['self.optimizer_state_dict'])
            if 'self.scheduler_state_dict' in checkpoint:
                assert self.scheduler is not None
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch_start = checkpoint['epoch']
            sub_epoch = checkpoint['sub_epoch'] + 1
            train_loss_all = checkpoint['train_loss']
            dev_loss_all = checkpoint["dev_loss"]
            test_loss_all = checkpoint["test_loss"]
            train_isdiv_all = checkpoint['train_isdiv']
            dev_isdiv_all = checkpoint["dev_isdiv"]
            test_isdiv_all = checkpoint["test_isdiv"]
            if 'extra_loss' in checkpoint:
                extra_loss_all = checkpoint['extra_loss']
                extra_isdiv_all = checkpoint['extra_isdiv']
            assert len(train_loss_all) == len(dev_loss_all) == len(test_loss_all) == len(train_isdiv_all) == len(dev_isdiv_all) == len(test_isdiv_all)
            for i, train_loss in enumerate(train_loss_all):
                if 'extra_loss' in checkpoint:
                    self.writer.add_scalars('Loss', {'train': train_loss, 'dev': dev_loss_all[i], 'test': test_loss_all[i], 'extra': extra_loss_all[i]}, i)
                    self.writer.add_scalars('IS Divergence', {'train': train_isdiv_all[i], 'dev': dev_isdiv_all[i], 'test': test_isdiv_all[i], 'extra': extra_isdiv_all[i]}, i)
                else:
                    self.writer.add_scalars('Loss', {'train': train_loss, 'dev': dev_loss_all[i], 'test': test_loss_all[i]}, i)
                    self.writer.add_scalars('IS Divergence', {'train': train_isdiv_all[i], 'dev': dev_isdiv_all[i], 'test': test_isdiv_all[i]}, i)
            tmp = "Resume training from epoch {} and subepoch {}\n".format(epoch_start, sub_epoch)
            self.logger.info(tmp)
            previous_checkpoint = int(os.path.basename(self.cfgs.path.resume).split('--', 1)[1].rsplit('.', 1)[0])
        else:
            """ Training from beginning """
            epoch_start, sub_epoch = 0, 1
            train_loss_all, dev_loss_all, test_loss_all = [], [], []
            train_isdiv_all, dev_isdiv_all, test_isdiv_all = [], [], []
            previous_checkpoint = -1
            if self.extra_loader is not None:
                extra_loss_all, extra_isdiv_all = [], []

        messages = []
        for expr, strategy in self.cfgs.opt_strategy.dropout_strategy.items():
            strategy = parse_dropout_strategy(strategy)
            dropout_rate = cal_drop_p(sub_epoch - 1, strategy)
            exec(expr)
            messages.append(f"Dropout rate for {expr.split('=')[0].strip()} is {dropout_rate}")
        messages.append("Learning rate is {:.7f}".format(self.optimizer.param_groups[0]['lr']))
        for mes in messages:
            print(mes)
            self.logger.info(mes)

        step = 0
        train_loss = 0.0
        train_isdiv = 0.0
        """ Training procedure """
        for epoch in range(epoch_start, self.cfgs.train.total_epoch_num):
            for in_data in self.train_loader:
                with self.interact as cmd:
                    if len(cmd) > 0:
                        for s in cmd:
                            self.cmd_dict[s]()
                        self.logger.info('Cmd: {} has been executed'.format(repr(cmd)))

                loss, is_div, grad_norm, nan_flag = self.TrainOneIteration(in_data)
                if nan_flag:
                    self.writer.close()
                    return nan_flag, previous_checkpoint
                train_loss += loss
                train_isdiv += is_div
                step += 1
                if step % self.cfgs.train.logging_interval == 0:
                    mesg = "Train SubEpoch:{}[{}/{}], Loss:{:.4e}, ISDiv: {:.4e}, GradNorm: {:.2e}".format(
                        sub_epoch, step, self.cfgs.train.validate_interval, train_loss / step, train_isdiv / step, grad_norm)
                    print(time.ctime() + ': ' + mesg)
                    self.logger.info(mesg)
                if step % self.cfgs.train.validate_interval == 0:
                    train_loss = train_loss / step
                    train_isdiv = train_isdiv / step
                    dev_loss, dev_isdiv = self.Validate(self.dev_loader, database='Dev')
                    test_loss, test_isdiv = self.Validate(self.test_loader, database='Test')
                    if self.extra_loader is not None:
                        extra_loss, extra_isdiv = self.Validate(self.extra_loader, database='Extra')

                    train_loss_all.append(train_loss)
                    dev_loss_all.append(dev_loss)
                    test_loss_all.append(test_loss)
                    train_isdiv_all.append(train_isdiv)
                    dev_isdiv_all.append(dev_isdiv)
                    test_isdiv_all.append(test_isdiv)
                    if self.extra_loader is None:
                        self.writer.add_scalars('Loss', {'train': train_loss, 'dev': dev_loss, 'test': test_loss}, sub_epoch)
                        self.writer.add_scalars('IS Divergence', {'train': train_isdiv, 'dev': dev_isdiv, 'test': test_isdiv}, sub_epoch)
                    else:
                        extra_loss_all.append(extra_loss)
                        extra_isdiv_all.append(extra_isdiv)
                        self.writer.add_scalars('Loss', {'train': train_loss, 'dev': dev_loss, 'test': test_loss, 'extra': extra_loss}, sub_epoch)
                        self.writer.add_scalars('IS Divergence', {'train': train_isdiv, 'dev': dev_isdiv, 'test': test_isdiv, 'extra': extra_isdiv}, sub_epoch)

                    if sub_epoch % self.cfgs.train.save_stdt_interval == 0:
                        torch.save(self.model.state_dict(), os.path.join(self.cfgs.path.model, 'state_dict--sub_epoch={}.pt'.format(sub_epoch)))
                    if sub_epoch % self.cfgs.train.save_ckpt_interval == 0:
                        previous_checkpoint = sub_epoch
                        ckpt_dict = {
                            'epoch': epoch,
                            'sub_epoch': sub_epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'train_loss': train_loss_all,
                            'dev_loss': dev_loss_all,
                            'test_loss': test_loss_all,
                            'train_isdiv': train_isdiv_all,
                            'dev_isdiv': dev_isdiv_all,
                            'test_isdiv': test_isdiv_all,
                        }
                        if self.extra_loader is not None:
                            ckpt_dict['extra_loss'] = extra_loss_all
                            ckpt_dict['extra_isdiv'] = extra_isdiv_all
                        if self.scheduler is not None:
                            ckpt_dict['self.scheduler_state_dict'] = self.scheduler.state_dict()
                        torch.save(ckpt_dict, os.path.join(self.cfgs.path.model, 'resume_model--{}.pt'.format(sub_epoch)))
                    sub_epoch += 1
                    train_loss = 0.0
                    train_isdiv = 0.0
                    step = 0

                    """ Apply Learning rate decay and dropout """
                    messages = []
                    if self.cfgs.opt_strategy.use_lr_decay and self.optimizer.param_groups[0]['lr'] > self.cfgs.opt_strategy.min_lr:
                        if (self.cfgs.opt_strategy.lr_activate_idx is not None and sub_epoch >= self.cfgs.opt_strategy.lr_activate_idx) or self.cfgs.opt_strategy.lr_activate_idx is None:
                            self.scheduler.step()
                    for expr, strategy in self.cfgs.opt_strategy.dropout_strategy.items():
                        strategy = parse_dropout_strategy(strategy)
                        dropout_rate = cal_drop_p(sub_epoch - 1, strategy)
                        exec(expr)
                        messages.append(f"Dropout rate for {expr.split('=')[0].strip()} is {dropout_rate}")
                    messages.append("Learning rate is {:.7f}".format(self.optimizer.param_groups[0]['lr']))
                    for mes in messages:
                        print(mes)
                        self.logger.info(mes)

        self.writer.close()
        return False, None

    def TrainNewModel(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.random.manual_seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        nan_cnt = 0

        """ Prepare dataloader """
        if self.cfgs.info.stage == 1:
            from dataset import CleanDataset
            self.transform = Spectrogram(**self.cfgs.sigproc)
            train_dataset = CleanDataset(self.cfgs.path.train, self.cfgs.dataset.usecols_egs, self.cfgs.general.data_prefix)
            dev_dataset = CleanDataset(self.cfgs.path.dev, self.cfgs.dataset.usecols_egs, self.cfgs.general.data_prefix)
            test_dataset = CleanDataset(self.cfgs.path.test, self.cfgs.dataset.usecols_egs, self.cfgs.general.data_prefix)
            train_sampler = TrainBatchSampler.from_dataset(train_dataset, self.cfgs.dataloader.batch_size, n_batch=self.cfgs.dataloader.nbatch_train, drop_last=self.cfgs.dataloader.droplast_train)
            dev_sampler = TrainBatchSampler.from_dataset(dev_dataset, self.cfgs.dataloader.batch_size, n_batch=self.cfgs.dataloader.nbatch_dev, drop_last=self.cfgs.dataloader.droplast_dev)
            test_sampler = TrainBatchSampler.from_dataset(test_dataset, self.cfgs.dataloader.batch_size, n_batch=self.cfgs.dataloader.nbatch_test, drop_last=self.cfgs.dataloader.droplast_test)
            self.train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=self.cfgs.dataloader.num_workers, collate_fn=CollateFnWrapper, pin_memory=True)
            self.dev_loader = DataLoader(dev_dataset, batch_sampler=dev_sampler, num_workers=self.cfgs.dataloader.num_workers, collate_fn=CollateFnWrapper, pin_memory=True)
            self.test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=self.cfgs.dataloader.num_workers, collate_fn=CollateFnWrapper, pin_memory=True)
            self.extra_loader = None
        elif self.cfgs.info.stage == 2:
            from dataset import NoisyDataset
            self.transform = Transform(self.cfgs.sigproc)
            extra_func = {'mix': self.cfgs.dataset.CreateMixWeight}
            train_dataset = NoisyDataset(self.cfgs.path.train, self.cfgs.dataset.usecols_ark, self.cfgs.dataset.usecols_egs, self.cfgs.general.prefix, self.cfgs.general.data_prefix, extra_func=extra_func)
            dev_dataset = NoisyDataset(self.cfgs.path.dev, self.cfgs.dataset.usecols_ark, self.cfgs.dataset.usecols_egs, self.cfgs.general.prefix, self.cfgs.general.data_prefix, extra_func=extra_func)
            test_dataset = NoisyDataset(self.cfgs.path.test, self.cfgs.dataset.usecols_ark, self.cfgs.dataset.usecols_egs, self.cfgs.general.prefix, self.cfgs.general.data_prefix, extra_func=extra_func)
            extra_dataset = NoisyDataset(self.cfgs.path.extra, self.cfgs.dataset.usecols_ark, self.cfgs.dataset.usecols_egs, self.cfgs.general.prefix, self.cfgs.general.data_prefix, extra_func=extra_func)
            self.train_loader = DataLoader(train_dataset, batch_size=self.cfgs.dataloader.batch_size, num_workers=self.cfgs.dataloader.num_workers,
                                           drop_last=self.cfgs.dataloader.droplast_train, pin_memory=True, shuffle=True, collate_fn=CollateFnWrapper)
            self.dev_loader = DataLoader(dev_dataset, batch_size=self.cfgs.dataloader.batch_size, num_workers=self.cfgs.dataloader.num_workers,
                                         drop_last=self.cfgs.dataloader.droplast_dev, pin_memory=True, shuffle=True, collate_fn=CollateFnWrapper)
            self.test_loader = DataLoader(test_dataset, batch_size=self.cfgs.dataloader.batch_size, num_workers=self.cfgs.dataloader.num_workers,
                                          drop_last=self.cfgs.dataloader.droplast_test, pin_memory=True, shuffle=True, collate_fn=CollateFnWrapper)
            self.extra_loader = DataLoader(extra_dataset, batch_size=self.cfgs.dataloader.batch_size, num_workers=self.cfgs.dataloader.num_workers,
                                           drop_last=self.cfgs.dataloader.droplast_extra, pin_memory=True, shuffle=True, collate_fn=CollateFnWrapper)
        mesg = [
            'Number of speakers in the training set: {}'.format(train_dataset.spkr_num),
            "Train dataloader length: {}".format(int(len(self.train_loader))),
            'Develop dataloader length: {}'.format(int(len(self.dev_loader))),
            'Test dataloader length: {}'.format(int(len(self.test_loader))),
        ]
        for mes in mesg:
            print(mes)
            if not self.cfgs.train.resume:
                self.logger.info(mes)

        """ Prepare self.model, self.loss, self.optimizer and self.scheduler """
        self.model = AutoEncoder(self.cfgs).to(self.cfgs.general.device)
        if self.cfgs.info.stage == 1:
            self.loss_class = Loss(self.cfgs.loss.lambda_rec, self.cfgs.loss.lambda_kl, loss_type=self.cfgs.loss.loss_type)
        elif self.cfgs.info.stage == 2:
            self.loss_class = Loss(self.cfgs.loss.lambda_rec, self.cfgs.loss.lambda_kl, self.cfgs.loss.freq_weight, loss_type=self.cfgs.loss.loss_type)
            checkpoint = torch.load(self.cfgs.path.stage_one, map_location=self.cfgs.general.device)
            self.model.load_state_dict(checkpoint)
            if self.cfgs.info.activate_content_encoder is False:
                for param in self.model.content_encoder.parameters():
                    param.requires_grad = False
            if self.cfgs.info.activate_decoder is False:
                for param in self.model.decoder.parameters():
                    param.requires_grad = False
            if self.cfgs.info.activate_speaker_encoder is False:
                for param in self.model.speaker_encoder.parameters():
                    param.requires_grad = False
        self.optimizer = eval(self.cfgs.opt_strategy.optim_strategy)
        if self.cfgs.opt_strategy.use_lr_decay:
            lr_decay_coeff = (self.cfgs.opt_strategy.min_lr / self.cfgs.opt_strategy.lr)**(1 / (self.cfgs.opt_strategy.lr_deactivate_idx - self.cfgs.opt_strategy.lr_activate_idx))
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, lr_decay_coeff)
            mesg = 'lr decay coefficient: {}\n'.format(lr_decay_coeff)
            print(mesg)
            self.logger.info(mesg)
        else:
            self.scheduler = None

        """ Start training new self.model """
        if not self.cfgs.train.resume:
            # Train from beginning
            self.resume_path = None
            self.tensorboard_title = self.cfgs.train.tensorboard_title
            is_nan, valid_subepoch = self.Train()
        else:
            # Train from checkpoint
            self.resume_path = self.cfgs.path.resume
            self.tensorboard_title = self.cfgs.train.tensorboard_title + '_resumefrom--{}'.format(os.path.basename(self.cfgs.path.resume).split('--', 1)[1].rsplit('.', 1)[0])
            is_nan, valid_subepoch = self.Train()
        while is_nan:
            if self.cfgs.train.nan_backroll:
                nan_cnt += 1
                if nan_cnt > self.cfgs.train.max_nan_allowance:
                    message = "LOSS TURNS NAN FOR TOO MANY TIMES. THEREFORE ABORT TRAINING!"
                    print(message)
                    self.logger.info(message)
                    break
                else:
                    assert valid_subepoch is not None
                    if valid_subepoch == -1:
                        message = "NAN APPEARS BEFROE THE FRIST SAVING POINT. THEREFORE ABORT TRAINING!"
                        print(message)
                        self.logger.info(message)
                        break
                    else:
                        message = "LOSS TURNS NAN THE {} TIME. THEREFORE ROLLING BACK TO SUBEPOCH {}".format(nan_cnt, valid_subepoch)
                        print(message)
                        self.logger.info(message)
                        self.resume_path = os.path.join(self.cfgs.path.model, 'resume_model--{}.pt'.format(valid_subepoch))
                        self.tensorboard_title = self.cfgs.train.self.tensorboard_title + '_resumefrom-{}'.format(valid_subepoch)
                        is_nan, valid_subepoch = self.Train()
            else:
                message = "LOSS TURNS NAN. THEREFORE ABORT TRAINING!"
                print(message)
                self.logger.info(message)

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


def Train(cfgs, seed):
    Train_ins = Training(cfgs, seed)
    Train_ins.TrainNewModel()

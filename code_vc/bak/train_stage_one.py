import os
import time
import torch
import random
import logging
import numpy as np
import statistics as stat
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import cfgs
from network import AutoEncoder, Loss
from PreProcess.data_utils import Spectrogram
from dataset import Dataset, TrainBatchSampler
from PreProcess.utils import CollateFnWrapper, cal_drop_p, parse_dropout_strategy


def CheckData(cfgs, logger, seed):
    random.seed(seed)
    torch.random.manual_seed(seed)
    transform = Spectrogram(**cfgs.sigproc)

    train_dataset = Dataset.from_pkl(cfgs.path.train, cfgs.general.data_prefix)
    vad_dataset = Dataset.from_pkl(cfgs.path.vad, cfgs.general.data_prefix)
    diagnosis_dataset = Dataset.from_pkl(cfgs.path.diagnosis, cfgs.general.data_prefix)

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


def Validating(logger, dataloader, model, transform, loss_class, device, database='Dev', logging_interval=30):
    model.eval()
    total_loss = 0.0
    total_isdiv = 0.0
    with torch.no_grad():
        for i, in_data in enumerate(dataloader):
            data = in_data.data.to(device)
            data, _, _ = transform(data)
            spkr_embd, content_mu, content_log_var, x_log_var = model(data)
            loss, is_div = loss_class(data, content_mu, content_log_var, x_log_var)
            total_loss += loss.item()
            total_isdiv += is_div.item()
            if (i + 1) % logging_interval == 0 or (i + 1) == len(dataloader):
                mesg = database + "Step[{}/{}], Loss:{:.4f}, ISDiv: {:.4f}".format(i + 1, len(dataloader), total_loss / (i + 1), total_isdiv / (i + 1))
                print(time.ctime() + ': ' + mesg)
                logger.info(mesg)
        return total_loss / (i + 1), total_isdiv / (i + 1)


def TrainOneIteration(in_data, model, optimizer, transform, loss_class, device,
                      use_grad_clip=True, grad_clip_strategy=None, pause_when_nan=True):
    nan_flag = False
    model.train()
    data = in_data.data.to(device)
    data, _, _ = transform(data)
    spkr_embd, content_mu, content_log_var, x_log_var = model(data)
    loss, is_div = loss_class(data, content_mu, content_log_var, x_log_var)
    """ Check for NAN """
    if torch.isnan(loss).any():
        nan_flag = True
        if pause_when_nan:
            import ipdb; ipdb.set_trace()
    """ Back propagation """
    optimizer.zero_grad()
    loss.backward()
    if use_grad_clip:
        grad_norm = eval(grad_clip_strategy)
    else:
        grad_norm = -1.
    optimizer.step()
    return loss.item(), is_div.item(), grad_norm, nan_flag


def Training(logger, model, train_loader, dev_loader, test_loader, epochs, optimizer, transform, loss_class, device,
             use_lr_decay=False, min_lr=None, lr_activate_idx=None, scheduler=None, dropout_strategy=None, use_grad_clip=True, grad_clip_strategy=None,
             logging_interval=30, validate_interval=360, save_stdt_interval=10, save_ckpt_interval=10,
             model_path=None, resume_path=None, tensorboard_title='run',
             pause_when_nan=True):

    if resume_path is None and os.path.exists(os.path.join(model_path, tensorboard_title)):
        os.system(f"rm -r {os.path.join(model_path, tensorboard_title)}")
    writer = SummaryWriter(os.path.join(model_path, tensorboard_title))
    if resume_path is not None:
        """ Training from checkpoint """
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            assert scheduler is not None
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_start = checkpoint['epoch']
        sub_epoch = checkpoint['sub_epoch'] + 1
        train_loss_all = checkpoint['train_loss']
        dev_loss_all = checkpoint["dev_loss"]
        test_loss_all = checkpoint["test_loss"]
        train_isdiv_all = checkpoint['train_isdiv']
        dev_isdiv_all = checkpoint["dev_isdiv"]
        test_isdiv_all = checkpoint["test_isdiv"]
        assert len(train_loss_all) == len(dev_loss_all) == len(test_loss_all) == len(train_isdiv_all) == len(dev_isdiv_all) == len(test_isdiv_all)
        for i, train_loss in enumerate(train_loss_all):
            writer.add_scalars('Loss', {'train': train_loss, 'dev': dev_loss_all[i], 'test': test_loss_all[i]}, i)   
            writer.add_scalars('IS Divergence', {'train': train_isdiv_all[i], 'dev': dev_isdiv_all[i], 'test': test_isdiv_all[i]}, i)   
        tmp = "Resume training from epoch {} and subepoch {}\n".format(epoch_start, sub_epoch)
        logger.info(tmp)
        previous_checkpoint = int(os.path.basename(cfgs.path.resume).split('--', 1)[1].rsplit('.', 1)[0])
    else:
        """ Training from beginning """
        epoch_start, sub_epoch = 0, 1
        train_loss_all, dev_loss_all, test_loss_all = [], [], []
        train_isdiv_all, dev_isdiv_all, test_isdiv_all = [], [], []
        previous_checkpoint = -1

    messages = []
    for expr, strategy in dropout_strategy.items():
        strategy = parse_dropout_strategy(strategy)
        dropout_rate = cal_drop_p(sub_epoch - 1, strategy)
        exec(expr)
        messages.append(f"Dropout rate for {expr.split('=')[0].strip()} is {dropout_rate}")
    messages.append("Learning rate is {:.7f}".format(optimizer.param_groups[0]['lr']))
    for mes in messages:
        print(mes)
        logger.info(mes)

    step = 0
    train_loss = 0.0
    train_isdiv = 0.0
    """ Training procedure """
    for epoch in range(epoch_start, epochs):
        for in_data in train_loader:
            loss, is_div, grad_norm, nan_flag = TrainOneIteration(in_data, model, optimizer, transform, loss_class, device, 
                                                                    use_grad_clip=use_grad_clip, grad_clip_strategy=grad_clip_strategy,
                                                                    pause_when_nan=pause_when_nan)
            if nan_flag:
                writer.close()
                return nan_flag, previous_checkpoint
            train_loss += loss
            train_isdiv += is_div
            step += 1
            if step % logging_interval == 0:
                mesg = "Train SubEpoch:{}[{}/{}], Loss:{:.4f}, ISDiv: {:.4f}, GradNorm: {:.2f}".format(
                    sub_epoch, step, validate_interval, train_loss / step, train_isdiv / step, grad_norm)
                print(time.ctime() + ': ' + mesg)
                logger.info(mesg)
            if step % validate_interval == 0:
                train_loss = train_loss / step
                train_isdiv = train_isdiv / step
                dev_loss, dev_isdiv = Validating(logger, dev_loader, model, transform, loss_class, device, database='Dev', logging_interval=logging_interval)
                test_loss, test_isdiv = Validating(logger, test_loader, model, transform, loss_class, device, database='Test', logging_interval=logging_interval)
                train_loss_all.append(train_loss)
                dev_loss_all.append(dev_loss)
                test_loss_all.append(test_loss)
                train_isdiv_all.append(train_isdiv)
                dev_isdiv_all.append(dev_isdiv)
                test_isdiv_all.append(test_isdiv)
                writer.add_scalars('Loss', {'train': train_loss, 'dev': dev_loss, 'test': test_loss}, sub_epoch)
                writer.add_scalars('IS Divergence', {'train': train_isdiv, 'dev': dev_isdiv, 'test': test_isdiv}, sub_epoch)
                if sub_epoch % save_stdt_interval == 0:
                    torch.save(model.state_dict(), os.path.join(model_path, 'state_dict--sub_epoch={}.pt'.format(sub_epoch)))
                if sub_epoch % save_ckpt_interval == 0:
                    previous_checkpoint = sub_epoch
                    ckpt_dict = {
                        'epoch': epoch,
                        'sub_epoch': sub_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss_all,
                        'dev_loss': dev_loss_all,
                        'test_loss': test_loss_all,
                        'train_isdiv': train_isdiv_all,
                        'dev_isdiv': dev_isdiv_all,
                        'test_isdiv': test_isdiv_all,
                    }
                    if scheduler is not None:
                        ckpt_dict['scheduler_state_dict'] = scheduler.state_dict()
                    torch.save(ckpt_dict, os.path.join(model_path, 'resume_model--{}.pt'.format(sub_epoch)))
                sub_epoch += 1
                train_loss = 0.0
                step = 0

                """ Apply Learning rate decay and dropout """
                messages = []
                if use_lr_decay and optimizer.param_groups[0]['lr'] > min_lr:
                    if (lr_activate_idx is not None and sub_epoch >= lr_activate_idx) or lr_activate_idx is None:
                        scheduler.step()
                for expr, strategy in dropout_strategy.items():
                    strategy = parse_dropout_strategy(strategy)
                    dropout_rate = cal_drop_p(sub_epoch - 1, strategy)
                    exec(expr)
                    messages.append(f"Dropout rate for {expr.split('=')[0].strip()} is {dropout_rate}")
                messages.append("Learning rate is {:.7f}".format(optimizer.param_groups[0]['lr']))
                for mes in messages:
                    print(mes)
                    logger.info(mes)

    writer.close()
    return False, None


def TrainNewModel(cfgs, logger, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    nan_cnt = 0

    """ Prepare dataloader """
    transform = Spectrogram(**cfgs.sigproc)

    train_dataset = Dataset(cfgs.path.train, cfgs.dataset.usecols, cfgs.general.data_prefix)
    dev_dataset = Dataset(cfgs.path.dev, cfgs.dataset.usecols, cfgs.general.data_prefix)
    test_dataset = Dataset(cfgs.path.test, cfgs.dataset.usecols, cfgs.general.data_prefix)

    train_sampler = TrainBatchSampler.from_dataset(train_dataset, cfgs.dataloader.batch_size, n_batch=cfgs.dataloader.nbatch_train, drop_last=cfgs.dataloader.droplast_train)
    dev_sampler = TrainBatchSampler.from_dataset(dev_dataset, cfgs.dataloader.batch_size, n_batch=cfgs.dataloader.nbatch_dev, drop_last=cfgs.dataloader.droplast_dev)
    test_sampler = TrainBatchSampler.from_dataset(test_dataset, cfgs.dataloader.batch_size, n_batch=cfgs.dataloader.nbatch_test, drop_last=cfgs.dataloader.droplast_test)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=cfgs.dataloader.num_workers, collate_fn=CollateFnWrapper, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_sampler=dev_sampler, num_workers=cfgs.dataloader.num_workers, collate_fn=CollateFnWrapper, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=cfgs.dataloader.num_workers, collate_fn=CollateFnWrapper, pin_memory=True)

    mesg = [
        'Number of speakers in the training set: {}'.format(train_dataset.spkr_num),
        "Train dataloader length: {}".format(int(len(train_loader))),
        'Develop dataloader length: {}'.format(int(len(dev_loader))),
        'Test dataloader length: {}'.format(int(len(test_loader))),
    ]
    for mes in mesg:
        print(mes)
        if not cfgs.train.resume:
            logger.info(mes)

    """ Prepare model, loss, optimizer and scheduler """
    model = AutoEncoder(cfgs).to(cfgs.general.device)
    loss_class = Loss(cfgs.loss.lambda_rec, cfgs.loss.lambda_kl)
    optimizer = eval(cfgs.opt_strategy.optim_strategy)
    if cfgs.opt_strategy.use_lr_decay:
        lr_decay_coeff = (cfgs.opt_strategy.min_lr / cfgs.opt_strategy.lr)**(1 / (cfgs.opt_strategy.lr_deactivate_idx - cfgs.opt_strategy.lr_activate_idx))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay_coeff)
        mesg = 'lr decay coefficient: {}\n'.format(lr_decay_coeff)
        print(mesg)
        logger.info(mesg)
    else:
        scheduler = None

    """ Start training new model """
    if not cfgs.train.resume:
        # Train from beginning
        is_nan, valid_subepoch = Training(logger, model, train_loader, dev_loader, test_loader, cfgs.train.total_epoch_num, optimizer, transform, loss_class, cfgs.general.device,
                                          use_lr_decay=cfgs.opt_strategy.use_lr_decay, min_lr=cfgs.opt_strategy.min_lr, lr_activate_idx=cfgs.opt_strategy.lr_activate_idx, scheduler=scheduler,
                                          dropout_strategy=cfgs.opt_strategy.dropout_strategy,
                                          use_grad_clip=cfgs.opt_strategy.use_grad_clip, grad_clip_strategy=cfgs.opt_strategy.grad_clip_strategy,
                                          logging_interval=cfgs.train.logging_interval, validate_interval=cfgs.train.validate_interval,
                                          save_stdt_interval=cfgs.train.save_stdt_interval, save_ckpt_interval=cfgs.train.save_ckpt_interval,
                                          model_path=cfgs.path.model, resume_path=None, tensorboard_title=cfgs.train.tensorboard_title,
                                          pause_when_nan=cfgs.train.pause_when_nan)
    else:
        # Train from checkpoint
        tensorboard_title = cfgs.train.tensorboard_title + '_resumefrom--{}'.format(os.path.basename(cfgs.path.resume).split('--', 1)[1].rsplit('.', 1)[0])
        is_nan, valid_subepoch = Training(logger, model, train_loader, dev_loader, test_loader, cfgs.train.total_epoch_num, optimizer, transform, loss_class, cfgs.general.device,
                                          use_lr_decay=cfgs.opt_strategy.use_lr_decay, min_lr=cfgs.opt_strategy.min_lr, lr_activate_idx=cfgs.opt_strategy.lr_activate_idx, scheduler=scheduler,
                                          dropout_strategy=cfgs.opt_strategy.dropout_strategy,
                                          use_grad_clip=cfgs.opt_strategy.use_grad_clip, grad_clip_strategy=cfgs.opt_strategy.grad_clip_strategy,
                                          logging_interval=cfgs.train.logging_interval, validate_interval=cfgs.train.validate_interval,
                                          save_stdt_interval=cfgs.train.save_stdt_interval, save_ckpt_interval=cfgs.train.save_ckpt_interval,
                                          model_path=cfgs.path.model, resume_path=cfgs.path.resume, tensorboard_title=tensorboard_title,
                                          pause_when_nan=cfgs.train.pause_when_nan)
    while is_nan:
        if cfgs.train.nan_backroll:
            nan_cnt += 1
            if nan_cnt > cfgs.train.max_nan_allowance:
                message = "LOSS TURNS NAN FOR TOO MANY TIMES. THEREFORE ABORT TRAINING!"
                print(message)
                logger.info(message)
                break
            else:
                assert valid_subepoch is not None
                if valid_subepoch == -1:
                    message = "NAN APPEARS BEFROE THE FRIST SAVING POINT. THEREFORE ABORT TRAINING!"
                    print(message)
                    logger.info(message)
                    break
                else:
                    message = "LOSS TURNS NAN THE {} TIME. THEREFORE ROLLING BACK TO SUBEPOCH {}".format(nan_cnt, valid_subepoch)
                    print(message)
                    logger.info(message)
                    resume_path = os.path.join(cfgs.path.model, 'resume_model--{}.pt'.format(valid_subepoch))
                    tensorboard_title = cfgs.train.tensorboard_title + '_resumefrom-{}'.format(valid_subepoch)
                    is_nan, valid_subepoch = Training(logger, model, train_loader, dev_loader, test_loader, cfgs.train.total_epoch_num, optimizer, transform, cfgs.general.device,
                                                      use_lr_decay=cfgs.opt_strategy.use_lr_decay, min_lr=cfgs.opt_strategy.min_lr, lr_activate_idx=cfgs.opt_strategy.lr_activate_idx, scheduler=scheduler,
                                                      dropout_strategy=cfgs.opt_strategy.dropout_strategy,
                                                      use_grad_clip=cfgs.opt_strategy.use_grad_clip, grad_clip_strategy=cfgs.opt_strategy.grad_clip_strategy,
                                                      logging_interval=cfgs.train.logging_interval, validate_interval=cfgs.train.validate_interval,
                                                      save_stdt_interval=cfgs.train.save_stdt_interval, save_ckpt_interval=cfgs.train.save_ckpt_interval,
                                                      model_path=cfgs.path.model, resume_path=resume_path, tensorboard_title=tensorboard_title,
                                                      pause_when_nan=cfgs.train.pause_when_nan)
        else:
            message = "LOSS TURNS NAN. THEREFORE ABORT TRAINING!"
            print(message)
            logger.info(message)


def Train(cfgs, seed):
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
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(hd)
        for mes in cfgs.pre_log:
            print(mes)
            logger.info(mes)
        TrainNewModel(cfgs, logger, seed)

    elif cfgs.info.mode == 'checkdata':
        """ Data checking """
        assert cfgs.general.train is False
        hd = logging.FileHandler(os.path.join(cfgs.path.model, 'checkdata.log'), mode='w')
        formatter = logging.Formatter(fmt="{asctime}: {message}", datefmt="%Y-%m-%d %H:%M:%S", style='{')
        hd.setLevel(logging.INFO)
        hd.setFormatter(formatter)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(hd)
        CheckData(cfgs, logger, seed)

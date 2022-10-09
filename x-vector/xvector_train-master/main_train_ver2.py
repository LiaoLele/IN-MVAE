import os
import time
import torch
import random
import logging
import torchaudio
import numpy as np
import statistics as stat
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import cfgs
from network import net
from dataset import Dataset, TrainBatchSampler
from utils import CollateFnWrapper, cal_drop_p, parse_dropout_strategy
# import fnmatch
# import librosa
# from utils import my_spectrogram
# import statistics as stat
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker


def process_cfgs(cfgs):
    cfgs.path.model = os.path.join(cfgs.path.root, 'model')
    os.makedirs(cfgs.path.model, exist_ok=True)
    if cfgs.train.use_lr_decay:
        assert cfgs.train.min_lr is not None
    if cfgs.train.use_dropout:
        assert cfgs.train.dropout_strategy is not None
    if cfgs.train.nan_backroll:
        assert cfgs.train.max_nan_allowance is not None
    if cfgs.general.train:
        assert cfgs.general.checkdata is False
    if cfgs.general.checkdata:
        assert cfgs.general.train is False


def check_data():
    random.seed(1)
    torch.random.manual_seed(0)

    train_dataset = Dataset.from_pkl(cfgs.path.train, cfgs.general.data_prefix)
    vad_dataset = Dataset.from_pkl(cfgs.path.vad, cfgs.general.data_prefix)
    diagnosis_dataset = Dataset.from_pkl(cfgs.path.diagnosis, cfgs.general.data_prefix)

    train_sampler = TrainBatchSampler.from_dataset(train_dataset, cfgs.train.batch_size, n_batch=None, drop_last=False)
    vad_sampler = TrainBatchSampler.from_dataset(vad_dataset, cfgs.train.batch_size, n_batch=cfgs.train.nbatch_for_validation, drop_last=True)
    diagnosis_sampler = TrainBatchSampler.from_dataset(diagnosis_dataset, cfgs.train.batch_size, n_batch=cfgs.train.nbatch_for_validation, drop_last=True)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=CollateFnWrapper, pin_memory=True)
    vad_loader = DataLoader(vad_dataset, batch_sampler=vad_sampler, num_workers=0, collate_fn=CollateFnWrapper, pin_memory=True)
    diagnosis_loader = DataLoader(diagnosis_dataset, batch_sampler=diagnosis_sampler, num_workers=0, collate_fn=CollateFnWrapper, pin_memory=True)

    mesg = [
        'Number of speakers: {}'.format(train_dataset.spkr_num),
        "Train_loader length: {}".format(int(len(train_loader))), # 就是n_batch
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
        print('in_data.shape')
        print(in_data.shape)
        # uttinfo = in_data.uttinfo
        # data = transform(data.float())
        # data = data.float().cuda(device)
        # for idx, samp in enumerate(data):
        #     data_mean = torch.mean(samp.mean(dim=-1)).item()
        #     data_std = torch.mean(samp.std(dim=-1)).item()
        #     if data_std < 1:
        #         print("std smaller than 1")
        #         import ipdb; ipdb.set_trace()
        #     messages = "data_mean: {:.4f}, data_std: {:.4f}, uttinfo: {}".format(data_mean, data_std, uttinfo[idx])
        #     print(messages)
        #     logger.info(messages)
        #     data_mean_all.append(abs(data_mean))
        #     data_std_all.append(data_std)
    # messages = [
    #     "AVG_mean: {:.4f}, AVG_std: {:.4f}".format(stat.mean(data_mean_all), stat.mean(data_std_all)),
    #     "mean_min: {:.4f}, mean_max: {:.4f}, std_min: {:.4f}, std_max: {:.4f}".format(min(data_mean_all), max(data_mean_all), min(data_std_all), max(data_std_all)),
    #     "mean_min_idx: {}, mean_max_idx: {}, std_min_idx: {}, std_max_idx: {}".format(data_mean_all.index(min(data_mean_all)), data_mean_all.index(max(data_mean_all)),
    #                                                                                   data_std_all.index(min(data_std_all)), data_std_all.index(max(data_std_all)))
    # ]
    # for mess in messages:
    #     print(mess)
    #     logger.info(mess)


def validate_one_iteration(dataloader, model, database='Vad', logging_interval=30):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, in_data in enumerate(dataloader):
            data, label = in_data.data, in_data.label
            data = transform(data.float())
            data = data.float().cuda(device)
            label = label.long().cuda(device)
            data = (data - data.mean(dim=-1, keepdim=True)) / (data.std(dim=-1, keepdim=True))
            prob = model(data)
            loss = F.cross_entropy(prob, label, reduction='mean')
            total_loss += loss.item()
            if (i + 1) % logging_interval == 0: #90
                mesg = database + "Step[{}/{}], TLoss:{:.4f}".format(i + 1, len(dataloader), total_loss / (i + 1))
                print(time.ctime() + ': ' + mesg)
                logger.info(mesg)
        return total_loss / (i + 1)


def train_one_iteration(in_data, model, optimizer, pause_when_nan=True):
    global prob_ori
    global loss_ori
    global loss_all_ori
    global uttinfo_ori
    nan_flag = False
    model.train()
    data, label, uttinfo = in_data.data, in_data.label, in_data.uttinfo

    data = transform(data.float())
    data = data.float().cuda(device)
    with torch.no_grad():
        data = (data - data.mean(dim=-1, keepdim=True)) / (data.std(dim=-1, keepdim=True))
    label = label.long().cuda(device) 
    prob = model(data) 
    loss = F.cross_entropy(prob, label, reduction='mean')
    loss_all = F.cross_entropy(prob, label, reduction='none')
    """ Check for NAN """
    if torch.isnan(loss).any():
        nan_flag = True
        if pause_when_nan:
            import ipdb; ipdb.set_trace()
    prob_ori.append(prob.detach().cpu().clone())
    loss_ori.append(loss.item())
    loss_all_ori.append(loss_all.detach().cpu().clone())
    uttinfo_ori.append(uttinfo)
    if len(prob_ori) == 6:
        prob_ori.pop(0)
        loss_ori.pop(0)
        loss_all_ori.pop(0)
        uttinfo_ori.pop(0)
    """ Back propagation """
    optimizer.zero_grad()
    loss.backward()
    for name, par in model.named_parameters():
        if torch.isinf(par).any() and pause_when_nan:
            import ipdb; ipdb.set_trace()
        if torch.isnan(par).any() and pause_when_nan:
            import ipdb; ipdb.set_trace()
        if torch.isinf(par.grad).any() and pause_when_nan:
            import ipdb; ipdb.set_trace()
        if torch.isnan(par.grad).any() and pause_when_nan:
            import ipdb; ipdb.set_trace()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
    optimizer.step()
    return loss.item(), nan_flag


def train_main(model, train_loader, vad_loader, diagnosis_loader, epochs, optimizer,
               use_lr_decay=False, min_lr=None, lr_activate_idx=None, scheduler=None, use_dropout=False, dropout_strategy=None,
               logging_interval=30, validate_interval=360, save_statedict_interval=10, save_resumemodel_interval=10,
               model_path=None, resume_path=None, tensorboard_title='run',
               pause_when_nan=True):
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
        vad_loss_all = checkpoint["vad_loss"]
        diagnosis_loss_all = checkpoint["diagnosis_loss"]
        assert len(train_loss_all) == len(vad_loss_all) == len(diagnosis_loss_all)
        for i, train_loss in enumerate(train_loss_all):
            writer.add_scalars('Loss', {'train_loss': train_loss, 'vad_loss': vad_loss_all[i], 'diagnosis_loss': diagnosis_loss_all[i]}, i)   
        tmp = "Resume training from epoch {} and subepoch {}\n".format(epoch_start, sub_epoch)
        logger.info(tmp)
        previous_checkpoint = int(os.path.basename(cfgs.path.resume).split('--', 1)[1].rsplit('.', 1)[0])
    else:
        """ Training from beginning """
        epoch_start, sub_epoch = 0, 1
        train_loss_all, vad_loss_all, diagnosis_loss_all = [], [], []
        previous_checkpoint = -1

    if use_dropout:
        dropout_strategy = parse_dropout_strategy(dropout_strategy)
        drop_p = cal_drop_p(sub_epoch - 1, dropout_strategy)
        model.drop_p = drop_p
    else:
        model.drop_p = 0.0
    step = 0
    train_loss = 0.0
    messages = [
        "Learning rate is {:.7f}".format(optimizer.param_groups[0]['lr']),
        "Dropout probability is {:.3f}".format(model.drop_p),
    ]
    for mes in messages:
        print(mes)
        logger.info(mes)

    """ Training procedure """
    for epoch in range(epoch_start, epochs):
        for in_data in train_loader:
            loss, nan_flag = train_one_iteration(in_data, model, optimizer, pause_when_nan=pause_when_nan)
            if nan_flag:
                writer.close()
                return nan_flag, previous_checkpoint
            train_loss += loss
            step += 1
            if step % logging_interval == 0: #90
                mesg = "Train SubEpoch:{}[{}/{}], TLoss:{:.4f}, drop_p:{:.4f}".format(
                    sub_epoch, step, validate_interval, train_loss / step, model.drop_p)
                print(time.ctime() + ': ' + mesg)
                logger.info(mesg)
            if step % validate_interval == 0:
                train_loss = train_loss / step
                vad_loss = validate_one_iteration(vad_loader, model, database='Vad', logging_interval=logging_interval)
                diagnosis_loss = validate_one_iteration(diagnosis_loader, model, database='Diagnosis', logging_interval=logging_interval)
                train_loss_all.append(train_loss)
                vad_loss_all.append(vad_loss)
                diagnosis_loss_all.append(diagnosis_loss)
                writer.add_scalars('Loss', {'train_loss': train_loss, 'vad_loss': vad_loss, 'diagnosis_loss': diagnosis_loss}, sub_epoch)
                if sub_epoch % save_statedict_interval == 0: 
                    torch.save(model.state_dict(), os.path.join(model_path, 'state_dict--sub_epoch={}.pt'.format(sub_epoch)))
                if sub_epoch % save_resumemodel_interval == 0: 
                    previous_checkpoint = sub_epoch
                    torch.save({
                        'epoch': epoch,
                        'sub_epoch': sub_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': train_loss_all,
                        'vad_loss': vad_loss_all,
                        'diagnosis_loss': diagnosis_loss_all,
                    }, os.path.join(model_path, 'resume_model--{}.pt'.format(sub_epoch)))
                sub_epoch += 1
                train_loss = 0.0
                step = 0
                """ Apply Learning rate decay and dropout """
                if use_lr_decay and optimizer.param_groups[0]['lr'] > min_lr:
                    if (lr_activate_idx is not None and sub_epoch >= lr_activate_idx) or lr_activate_idx is None:
                        scheduler.step()
                if cfgs.train.use_dropout:
                    drop_p = cal_drop_p(sub_epoch - 1, dropout_strategy)
                    model.drop_p = drop_p
                messages = [
                    "Learning rate is {:.7f}".format(optimizer.param_groups[0]['lr']),
                    "Dropout probability is {:.3f}".format(model.drop_p),
                ]
                for mes in messages:
                    print(mes)
                    logger.info(mes)
    writer.close()
    return False, None


def train_new_model():
    random.seed(1)
    torch.random.manual_seed(0)
    nan_cnt = 0

    """ Prepare dataloader """
    train_dataset = Dataset.from_pkl(cfgs.path.train, cfgs.general.data_prefix)
    vad_dataset = Dataset.from_pkl(cfgs.path.vad, cfgs.general.data_prefix)
    diagnosis_dataset = Dataset.from_pkl(cfgs.path.diagnosis, cfgs.general.data_prefix)

    train_sampler = TrainBatchSampler.from_dataset(train_dataset, cfgs.train.batch_size, n_batch=None, drop_last=False)
    vad_sampler = TrainBatchSampler.from_dataset(vad_dataset, cfgs.train.batch_size, n_batch=cfgs.train.nbatch_for_validation, drop_last=True)
    diagnosis_sampler = TrainBatchSampler.from_dataset(diagnosis_dataset, cfgs.train.batch_size, n_batch=cfgs.train.nbatch_for_validation, drop_last=True)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=cfgs.train.num_workers, collate_fn=CollateFnWrapper, pin_memory=True)
    vad_loader = DataLoader(vad_dataset, batch_sampler=vad_sampler, num_workers=cfgs.train.num_workers, collate_fn=CollateFnWrapper, pin_memory=True)
    diagnosis_loader = DataLoader(diagnosis_dataset, batch_sampler=diagnosis_sampler, num_workers=cfgs.train.num_workers, collate_fn=CollateFnWrapper, pin_memory=True)

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

    """ Prepare model, optimizer and scheduler """
    model = net(train_dataset.spkr_num).cuda(device)
    optimizer = eval(cfgs.train.optim_strategy)  # TODO: sgd or adam
    if cfgs.train.use_lr_decay:
        lr_decay_coeff = (cfgs.train.min_lr / cfgs.train.lr)**(1 / (cfgs.train.lr_deactivate_idx - cfgs.train.lr_activate_idx))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay_coeff)
        mesg = 'lr decay coefficient: {}\n'.format(lr_decay_coeff)
        print(mesg)
        logger.info(mesg)
    else:
        scheduler = None

    """ Start training new model """
    if not cfgs.train.resume:
        # Train from beginning
        is_nan, valid_subepoch = train_main(model, train_loader, vad_loader, diagnosis_loader, cfgs.train.total_epoch_num, optimizer,
                                            use_lr_decay=cfgs.train.use_lr_decay, min_lr=cfgs.train.min_lr, lr_activate_idx=cfgs.train.lr_activate_idx, scheduler=scheduler,
                                            use_dropout=cfgs.train.use_dropout, dropout_strategy=cfgs.train.dropout_strategy,
                                            logging_interval=cfgs.train.logging_interval, validate_interval=cfgs.train.validate_interval,
                                            save_statedict_interval=cfgs.train.save_statedict_interval, save_resumemodel_interval=cfgs.train.save_resumemodel_interval,
                                            model_path=cfgs.path.model, resume_path=None, tensorboard_title=cfgs.train.tensorboard_title,
                                            pause_when_nan=cfgs.train.pause_when_nan)
    else:
        # Train from checkpoint
        tensorboard_title = cfgs.train.tensorboard_title + '_resumefrom--{}'.format(os.path.basename(cfgs.path.resume).split('--', 1)[1].rsplit('.', 1)[0])
        is_nan, valid_subepoch = train_main(model, train_loader, vad_loader, diagnosis_loader, cfgs.train.total_epoch_num, optimizer,
                                            use_lr_decay=cfgs.train.use_lr_decay, min_lr=cfgs.train.min_lr, lr_activate_idx=cfgs.train.lr_activate_idx, scheduler=scheduler,
                                            use_dropout=cfgs.train.use_dropout, dropout_strategy=cfgs.train.dropout_strategy,
                                            logging_interval=cfgs.train.logging_interval, validate_interval=cfgs.train.validate_interval,
                                            save_statedict_interval=cfgs.train.save_statedict_interval, save_resumemodel_interval=cfgs.train.save_resumemodel_interval,
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
                    is_nan, valid_subepoch = train_main(model, train_loader, vad_loader, diagnosis_loader, cfgs.train.total_epoch_num, optimizer,
                                                        use_lr_decay=cfgs.train.use_lr_decay, min_lr=cfgs.train.min_lr, lr_activate_idx=cfgs.train.lr_activate_idx, scheduler=scheduler,
                                                        use_dropout=cfgs.train.use_dropout, dropout_strategy=cfgs.train.dropout_strategy,
                                                        logging_interval=cfgs.train.logging_interval, validate_interval=cfgs.train.validate_interval,
                                                        save_statedict_interval=cfgs.train.save_statedict_interval, save_resumemodel_interval=cfgs.train.save_resumemodel_interval,
                                                        model_path=cfgs.path.model, resume_path=resume_path, tensorboard_title=tensorboard_title,
                                                        pause_when_nan=cfgs.train.pause_when_nan)
        else:
            message = "LOSS TURNS NAN. THEREFORE ABORT TRAINING!"
            print(message)
            logger.info(message)


if __name__ == "__main__":
    process_cfgs(cfgs)
    # device = torch.device("cuda: {}".format(cfgs.general.device))
    device = torch.device(cfgs.general.device)
    transform = torchaudio.transforms.MFCC(sample_rate=cfgs.sigproc.sr, n_mfcc=cfgs.model.feat_num, melkwargs=cfgs.mfcc)  # TODO: difference between melkwargs mfcc.dim and n_mfcc

    if cfgs.general.train:
        """ Training """
        assert cfgs.general.checkdata is False
        prob_ori = []
        loss_ori = []
        loss_all_ori = []
        uttinfo_ori = []
        if cfgs.train.resume is False:
            # backup network.py and conf.py
            os.system("cp {} {}".format("network.py", os.path.join(cfgs.path.root, 'network.py')))
            os.system("cp {} {}".format("conf.py", os.path.join(cfgs.path.root, 'config.txt')))
            # setting logger
            hd = logging.FileHandler(os.path.join(cfgs.path.root, 'train.log'), mode='w')
        else:
            hd = logging.FileHandler(os.path.join(cfgs.path.root, 'train.log'), mode='a')
            # mode='a'追加写入
        formatter = logging.Formatter(fmt="{asctime}: {message}", datefmt="%Y-%m-%d %H:%M:%S", style='{')
        hd.setLevel(logging.INFO)
        hd.setFormatter(formatter)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(hd)
        for mes in cfgs.pre_log:
            print(mes)
            logger.info(mes)
        train_new_model()
    if cfgs.general.checkdata:
        """ Data checking """
        assert cfgs.general.train is False
        hd = logging.FileHandler(os.path.join(cfgs.path.root, 'checkdata.log'), mode='w')
        formatter = logging.Formatter(fmt="{asctime}: {message}", datefmt="%Y-%m-%d %H:%M:%S", style='{')
        hd.setLevel(logging.INFO)
        hd.setFormatter(formatter)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(hd)
        check_data()

    # else:
    #     verification()
        # plot()
    # correction()


    # def plot():
    #     root = '.sigproc.hdd0/zhaoyigu/PROJECT/GE2E_speaker_encoder.sigproc.test_02/model/'
    #     legend = ['training.sigproc., 'test.sigproc.]
    #     txt_file = [
    #         root + 'Softmax Accuracy result for training set.txt',
    #         root + 'Softmax Accuracy result for test set.txt',
    #     ]
    #     total_acc = []
    #     for txt in txt_file:
    #         acc = []
    #         f = open(txt, 'r')
    #         while True:
    #             line = f.readline()
    #             if not line:
    #                 break
    #             epoch = int(line.rsplit('=')[1].rsplit('.')[0]) + 1
    #             acc.append((int(line[-5: -1]) * 1e-2, epoch))
    #         acc = sorted(acc, key=lambda x: x[1])
    #         acc, x_ticks = [x for x, _ in acc], [str(y) for _, y in acc]
    #         acc = np.array(acc)
    #         total_acc.append(acc)
    #     total_acc = np.stack(total_acc, 0)
    #     fig, ax = plt.subplots()
    #     ax.set_title('Accuracy rate')
    #     for i, acc in enumerate(total_acc):
    #         plt.plot(np.arange(1, acc.shape[0] + 1), acc, label=legend[i])

    #     ax.set_xlabel('epoch')
    #     ax.set_xticks(np.arange(1, 11))
    #     ax.set_xticklabels(x_ticks)
    #     ax.set_yticks(np.arange(95, 100.5, 0.5))
    #     ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    #     ax.set_ylabel('Accuracy rate [%]')
    #     ax.legend()
    #     plt.show()
    #     fig.savefig(os.path.join(root, 'accuracy rate.png'))



# def verification():

#     test.sigproc.et = my.sigproc.et_test(cfgs.test.test_path, cfgs.test.enroll_num, cfgs.test.verify_num, 
#                                    frame_num=cfgs.test.frame_num, enroll_hop=cfgs.test.enroll_hop,
#                                    verify_hop=cfgs.test.verify_hop, num_speaker=cfgs.test.num_speaker)
#     test_loader = DataLoader(test.sigproc.et, batch_size=cfgs.test.N, shuffle=True, num_workers=cfgs.test.num_workers, drop_last=True)
#     model = speaker_net().to(device)
#     model_file = list(filter(lambda x: fnmatch.fnmatch(x, 'state_dict*=*.pt'), os.listdir(cfgs.test.model_path)))
#     fb_mat = torch.from_numpy(librosa.filters.mel(cfgs.sigproc.sr, n_fft, n_mels=cfgs.sigproc.nmels)).unsqueeze(0)

#     for file in model_file:
#         state_dict = torch.load(os.path.join(cfgs.test.model_path, file), map_location=device)
#         model.load_state_dict(state_dict)
#         model.eval()

#         total_eer = 0.0
#         # total_acc_rate = 0.0
#         with torch.no_grad():
#             for epoch in range(cfgs.test.epochs):
#                 batch_eer = 0.0
#                 # batch_acc_rate = []
#                 for i,.sigproc.in enumerate(test_loader):
#                    .sigproc.=.sigproc.contiguous().view(-1,.sigproc.shape[-1]).to(device).float()
#                    .sigproc.= my_spectrogram.sigproc. n_fft, hop)
#                    .sigproc.= torch.matmul(fb_mat.cuda(device),.sigproc.
#                    .sigproc.= 10 * torch.log10(torch.clamp.sigproc. 1e-10))
#                    .sigproc.=.sigproc.view(cfgs.test.N, -1,.sigproc.shape[-2],.sigproc.shape[-1])
#                     enrollment, verification =.sigproc.:, 0:cfgs.test.enroll_num, :, :],.sigproc.:, cfgs.test.enroll_num:, :, :]  # enroll [batch, M/2, time, n_feature]; valid [batch, M/2, time, n_feature]
#                     enrollment = enrollment.contiguous().view(-1, enrollment.shape[-2], enrollment.shape[-1]).to(device).float()
#                     verification = verification.contiguous().view(-1, verification.shape[-2], verification.shape[-1]).to(device).float()

#                     enrollment_embeddings = model.embedding_layer(enrollment.permute(0, 2, 1))
#                     verification_embeddings = model.embedding_layer(verification.permute(0, 2, 1))
#                     enrollment_embeddings = enrollment_embeddings.view(cfgs.test.N, cfgs.test.enroll_num, -1)
#                     verification_embeddings = verification_embeddings.view(cfgs.test.N, cfgs.test.verify_num, -1)

#                     centroids = torch.mean(enrollment_embeddings, dim=1)  # [N, d_embedding]
#                     cosine_mat = get_cosmat(verification_embeddings, centroids, mode='inclusive')  # [N, verify_num, N]

#                     """ SOFTMAX """
#             #         softmax_mat = F.softmax(cosine_mat, dim=2)
#             #         _, label = torch.max(softmax_mat, dim=2, keepdim=True)
#             #         acc_rate = torch.sum(torch.stack([label[i, :, 0] == i for i in range(cfgs.test.N)])).float() / (cfgs.test.N * cfgs.test.verify_num)
#             #         batch_acc_rate.append(acc_rate.item())
#             #     print('epoch {} in file {}'.format(epoch + 1, file))
#             #     # total_acc_rate += batch_acc_rate / (i + 1)
#             #     # print("{} Softmax accurate rate is {:.4f} in epoch {}".format(file, batch_acc_rate / (i + 1), epoch + 1))
#             # total_acc_rate = stat.mean(batch_acc_rate)
#             # total_acc_std = stat.stdev(batch_acc_rate)
#             # print("\n{} Total softmax accurate rate is {:.4f}\nTotal softmax accurate std is {:.4f}\n".format(file, total_acc_rate, total_acc_std))
#             # with open(os.path.join(cfgs.test.model_path, 'Softmax Accuracy result for test set.txt'), 'a+') as f:
#             #     f.write("FILE: {2}---Total softmax accurate and std across {0} epochs: {1:.4f}\t{3:.4f}\n".format(cfgs.test.epochs, total_acc_rate, file, total_acc_std))
#             # f.close()
#             # if total_acc_rate > best_performance:
#             #     best_performance = total_acc_rate
#             #     torch.save(checkpoint['model_state_dict'], os.path.join(cfgs.test.model_path, 'best_state_dict--softmax.pt'))

#                     """ ERR """
#                     batch_eer = EER_calc(cosine_mat, batch_eer, file)
#                 total_eer += batch_eer / (i + 1)
#             total_eer = total_eer / cfgs.test.epochs
#             print("\n FILE: {2}---EER across {0} epochs: {1:.4f}\n".format(cfgs.test.epochs, total_eer, file))
#             with open(os.path.join(cfgs.test.model_path, 'EER_result.txt'), 'w') as f:
#                 f.write("FILE: {2}---EER across {0} epochs: {1:.4f}".format(cfgs.test.epochs, total_eer, file))
#             f.close()
#             # if total_eer < best_performance:
#             #     best_performance = total_acc_rate
#             #     torch.save(checkpoint['model_state_dict'], os.path.join(cfgs.test.model_path, 'best_state_dict--softmax.pt'))


# def correction():
#     writer = SummaryWriter(os.path.join(cfgs.model.model_path, cfgs.train.tensorboard_title))
#     checkpoint = torch.load(cfgs.train.resume.sigproc.path, map_location=device)
#     train.sigproc.et = my.sigproc.et(cfgs.sigproc.train_path, cfgs.train.M, cfgs.sigproc.train_start_time, cfgs.sigproc.train_set, cfgs.sigproc.train_spkrs, use_random_len=cfgs.train.use_random_len)
#     test.sigproc.et = tmp.sigproc.et(cfgs.sigproc.test_path, cfgs.train.M, cfgs.sigproc.test_start_time, cfgs.sigproc.test_set, cfgs.sigproc.test_spkrs, use_random_len=cfgs.train.use_random_len)
#     test_loader = DataLoader(test.sigproc.et, batch_size=cfgs.train.N, shuffle=True, num_workers=cfgs.train.num_workers,
#                              drop_last=True, pin_memory=False, collate_fn=my_collate_fn)
#     test_loss_correct = []
#     res = [5]

#     model = net(len(train.sigproc.et)).cuda(device)
#     model_file = list(filter(lambda x: x.startswith('state'), os.listdir('.sigproc.hdd0/zhaoyigu/PROJECT/Xvector_speaker_encoder.sigproc.test_voxceleb_03/model/')))
#     model_file = sorted(model_file, key=lambda x: int(x.split('=')[1].split('.')[0]))
#     for i, state_dict in enumerate(model_file):
#         model.load_state_dict(torch.load(os.path.join('.sigproc.hdd0/zhaoyigu/PROJECT/Xvector_speaker_encoder.sigproc.test_voxceleb_03/model/', state_dict), 
#                                          map_location=device))
#         test_loss = validation_module(test_loader, model, 0)
#         test_loss_correct.extend(list(np.linspace(res[-1], test_loss, num=99 if i == 0 else 100, endpoint=False)))
#         res.append(test_loss)

#     model.load_state_dict(checkpoint['model_state_dict'])
#     checkpoint['test_loss_correct'] = test_loss_correct
#     test_loss = validation_module(test_loader, model, 0)
#     test_loss_correct.extend(list(np.linspace(res[-1], test_loss, num=checkpoint['epoch'] - 1699 + 1, endpoint=True)))
#     res.append(test_loss)
#     assert len(test_loss_correct) == len(checkpoint['train_loss'])

#     torch.save({
#         'epoch': checkpoint['epoch'],
#         'model_state_dict': checkpoint['model_state_dict'],
#         'print_model': checkpoint['print_model'],
#         'optimizer_state_dict': checkpoint['optimizer_state_dict'],
#         'scheduler_state_dict': checkpoint['scheduler_state_dict'],
#         'train_loss': checkpoint['train_loss'],
#         'test_loss': checkpoint['test_loss'],
#         'test_loss_correct': test_loss_correct,
#     }, os.path.join(cfgs.model.model_path, 'model_new.pt'))  


# def EER_calc(cosine_mat, batch_eer, file):
#     diff = 1.0
#     for thold in [0.01 * i + 0.1 for i in range(50)]:
#         judge_positive = cosine_mat > thold
#         frr = hp.test.N * hp.test.verify_num - torch.sum(torch.stack([judge_positive[i, :, i] for i in range(hp.test.N)]))
#         far = judge_positive.sum() - torch.sum(torch.stack([judge_positive[i, :, i] for i in range(hp.test.N)]))
#         frr = frr.float() / (hp.test.N * hp.test.verify_num)
#         far = far.float() / (hp.test.N**2 * hp.test.verify_num - hp.test.N * hp.test.verify_num) 
#         err = abs(far - frr)
#         if err < diff:
#             diff = err
#             eer = (frr + far) / 2
#             eer_thresh = thold
#             eer_far = far
#             eer_frr = frr
#     batch_eer += eer
#     print('FILE: {4}  EER: {0:.2f}, FAR: {1:.2f}, FRR: {2:.2f}, threshold: {3:.2f}'.format(eer, eer_far, eer_frr, eer_thresh, file))
#     return batch_eer

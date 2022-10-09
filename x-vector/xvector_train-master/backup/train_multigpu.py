import torch
from dataset import my_dataset
from config.hparam import hparam as hp
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from network import net
import time
from torch.utils.tensorboard import SummaryWriter
import os
from utils import image, my_spectrogram, plot_spectrogram
import random
import argparse
import librosa
import numpy as np

import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.distributed as dist

from prefetch_generator import BackgroundGenerator


os.makedirs(hp.train.model_path, exist_ok=True)
n_fft = int(hp.data.stft_frame * hp.data.sr)
hop = int(hp.data.stft_hop * hp.data.sr)


def from_resume(model, optimizer, scheduler, epoch_start, current_gpu):
    checkpoint = torch.load(hp.train.resume_data_path, map_location='cuda:{}'.format(current_gpu))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch_start = checkpoint['epoch']
    return model, optimizer, scheduler, epoch_start


def train_module(dataloader, model, class_loss, optimizer, epoch, current_gpu):
    
    ret = {}
    model.train()
    total_loss, total_basic_loss, total_cycle_loss, total_is_div = 0.0, 0.0, 0.0, 0.0
    
    for i, data in enumerate(dataloader):
        data = data.float().cuda(current_gpu)

        data = my_spectrogram(data, n_fft, hop)
        mu, logvar, log_sigma, embeddings, embeddings_predict = model(data)
        final_loss, loss_basic, loss_cycle, is_div = class_loss(data, mu, logvar, log_sigma, embeddings, embeddings_predict)

        # data = prefetcher.next()

        model.zero_grad()
        optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        torch.distributed.all_reduce(final_loss, op=dist.ReduceOp.SUM)
        torch.distributed.all_reduce(loss_basic, op=dist.ReduceOp.SUM)
        torch.distributed.all_reduce(is_div, op=dist.ReduceOp.SUM)
        if loss_cycle is not None:
            torch.distributed.all_reduce(loss_cycle, op=dist.ReduceOp.SUM)

        if current_gpu == 0:
            final_loss.div_(dist.get_world_size()) 
            loss_basic.div_(dist.get_world_size())
            is_div.div_(dist.get_world_size())

            total_loss += final_loss.item()
            total_basic_loss += loss_basic.item()
            total_is_div += is_div.item()

            if loss_cycle is not None:
                loss_cycle.div_(dist.get_world_size())
                total_cycle_loss += loss_cycle.item()

            if (i + 1) % hp.train.log_interval == 0:
                mesg = "{0} Train Epoch:{1}[{2}/{3}], Loss:{4:.4f}, TLoss:{5:.4f}".format(time.ctime(), epoch + 1, i + 1, len(dataloader),
                                                                                          final_loss, total_loss / (i + 1))
                print(mesg)

    if current_gpu == 0:
        ret['Loss'] = total_loss / (i + 1)
        ret['Basic loss'] = total_basic_loss / (i + 1)
        ret['IS divergence'] = total_is_div / (i + 1)
        if loss_cycle is not None:
            ret['Cycle loss'] = total_cycle_loss / (i + 1)

    return ret


def validation_module(dataloader, model, class_loss, epoch, current_gpu):
    ret = {}
    model.eval()
    total_loss, total_basic_loss, total_cycle_loss, total_is_div = 0.0, 0.0, 0.0, 0.0

    # prefetcher = data_prefetcher(dataloader, current_gpu)
    # data = prefetcher.next()
    # i = -1
    with torch.no_grad():
    #     while data is not None:
    #         i += 1
        for i, data in enumerate(dataloader):

            data = data.float().cuda(current_gpu)
            data = my_spectrogram(data, n_fft, hop)
            mu, logvar, log_sigma, embeddings, embeddings_predict = model(data)
            final_loss, loss_basic, loss_cycle, is_div = class_loss(data, mu, logvar, log_sigma, embeddings, embeddings_predict)

            # data = prefetcher.next()

            torch.distributed.all_reduce(final_loss, op=dist.ReduceOp.SUM)
            torch.distributed.all_reduce(loss_basic, op=dist.ReduceOp.SUM)
            torch.distributed.all_reduce(is_div, op=dist.ReduceOp.SUM)
            if loss_cycle is not None:
                torch.distributed.all_reduce(loss_cycle, op=dist.ReduceOp.SUM)

            if current_gpu == 0:
                final_loss.div_(dist.get_world_size()) 
                loss_basic.div_(dist.get_world_size())
                is_div.div_(dist.get_world_size())

                total_loss += final_loss.item()
                total_basic_loss += loss_basic.item()
                total_is_div += is_div.item()

                if loss_cycle is not None:
                    loss_cycle.div_(dist.get_world_size())
                    total_cycle_loss += loss_cycle.item()
             
    if current_gpu == 0:
        mesg = "{0} Test Epoch:{1}[{2}/{3}], Loss:{4:.4f}, TLoss:{5:.4f}".format(time.ctime(), epoch + 1, i + 1, len(dataloader),
                                                                                 final_loss.item(), total_loss / (i + 1))
        print(mesg)

        ret['Loss'] = total_loss / (i + 1)
        ret['Basic loss'] = total_basic_loss / (i + 1)
        ret['IS divergence'] = total_is_div / (i + 1)
        if loss_cycle is not None:
            ret['Cycle loss'] = total_cycle_loss / (i + 1)

    return ret


def main_worker(ngpus, args):

    random.seed(42)
    torch.manual_seed(42)

    # muli-gpu configuration initialization
    dist.init_process_group('NCCL', world_size=ngpus, rank=args.local_rank)
    current_gpu = args.local_rank
    torch.cuda.set_device(current_gpu)
    print('Use gpu: {}'.format(current_gpu))

    # model initialization
    model = net().cuda(current_gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_gpu], output_device=current_gpu)

    # training strategy initialization
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.train.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hp.train.lr_decay_step, gamma=hp.train.lr_decay_coeff)
    epoch_start = 0

    # resume training checking
    if hp.train.resume:
        model, optimizer, scheduler, epoch_start = from_resume(model, optimizer, scheduler, epoch_start, current_gpu)
    
    cudnn.benchmark = True

    batch_size = int(hp.train.N / ngpus)
    num_workers = int(hp.train.num_workers / ngpus)
    train_dataset = my_dataset(hp.data.train_path, state='Train')
    test_dataset = my_dataset(hp.data.test_path, state='Test')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers,
        drop_last=False, pin_memory=True, sampler=train_sampler)  # 
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers,
        drop_last=False, pin_memory=True, sampler=test_sampler)

    if args.local_rank == 0:
        writer = SummaryWriter(os.path.join(hp.train.model_path, hp.train.tensorboard_title))
        plot_samples = torch.from_numpy(np.stack((train_dataset[0], test_dataset[40]), axis=0)).float().cuda(current_gpu)
        image(plot_samples, writer, mode='oracle')
        print(len(train_dataset))
        print(len(test_dataset))
    
    assert torch.distributed.is_initialized()
    dist.barrier()
    for epoch in range(epoch_start, hp.train.epochs):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        ret_train = train_module(train_loader, model, class_loss, optimizer, epoch, current_gpu)  #
        ret_test = validation_module(test_loader, model, class_loss, epoch, current_gpu)  # 

        # if optimizer.param_groups[0]['lr'] > hp.train.min_lr:
        #     scheduler.step()

        if args.local_rank == 0:
            if (epoch + 1) % hp.train.plot_interval == 0:
                image(plot_samples, writer, model=model, device='cuda:0', epoch=epoch + 1, mode='test')

            if (epoch + 1) % hp.train.checkpoint_interval == 0:
                torch.save(model.module.state_dict(), os.path.join(hp.train.model_path, 'state_dict--epoch={}.pt').format(epoch + 1))
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'print_model': model.state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(hp.train.model_path, 'model.pt'))     

            for label in ret_train:
                # writer.add_scalar(label, ret_train[label], epoch)
                writer.add_scalars(label, {'train': ret_train[label], 'test': ret_test[label]}, epoch + 1)  
    if args.local_rank == 0: 
        writer.close()


def main():
    # train_dataset = my_dataset(hp.data.test_path, state='Test')
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    os.environ['MASTER_PORT'] = '23456'
    os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2,3]))
    ngpus = 2
    # mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, ))
    main_worker(ngpus, args)


def check():
    # model = net()
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(hp.train.pretrained_path, map_location=torch.device('cpu'))
    # pretrained_dict_use = {}
    # for k, v in pretrained_dict.items():
    #     if k in model_dict:
    #         print(k)
    #         pretrained_dict_use[k] = v
    # model_dict.update(pretrained_dict_use)
    # model.load_state_dict(model_dict)
    # model_dict = model.state_dict()
    # for k, v in pretrained_dict_use.items():
    #     assert (model_dict[k].numpy() == v.numpy()).all()

    checkpoint = torch.load(hp.check.check_model_path, map_location='cpu')
    # for k, b in pretrained_dict_use.items():
    #     assert (checkpoint[k].numpy() == b.numpy()).all()

    model = cVAE(d_mel=128, n_embedding=90, enable_predict=False)

    model.load_state_dict(checkpoint)
    print('log_g')
    print(model.speech_model.log_g)
    print('encoder_conv1')
    print(model.speech_model.encoder_conv1.conv.weight.shape)
    print(model.speech_model.encoder_conv1.conv.weight[:, 0: 513, :].abs().sum() / 513)
    print(model.speech_model.encoder_conv1.conv.weight[:, 513:, :].abs().sum() / hp.model.embedding)
    print('encoder_conv2')
    print(model.speech_model.encoder_conv2.conv.weight.shape)
    print(model.speech_model.encoder_conv2.conv.weight[:, 0: hp.model.channels[0], :].abs().sum() / hp.model.channels[0])
    print(model.speech_model.encoder_conv2.conv.weight[:, hp.model.channels[0]:, :].abs().sum() / hp.model.embedding)
    print('encoder_mu')
    print(model.speech_model.encoder_mu.weight.shape)
    print(model.speech_model.encoder_mu.weight[:, 0: hp.model.channels[1], :].abs().sum() / hp.model.channels[1])
    print(model.speech_model.encoder_mu.weight[:, hp.model.channels[1]:, :].abs().sum() / hp.model.embedding)
    print('encoder_logvar')
    print(model.speech_model.encoder_logvar.weight.shape)
    print(model.speech_model.encoder_logvar.weight[:, 0: hp.model.channels[1], :].abs().sum() / hp.model.channels[1])
    print(model.speech_model.encoder_logvar.weight[:, hp.model.channels[1]:, :].abs().sum() / hp.model.embedding)
    # # for name, param in model.named_parameters():
    # #     print(name)




if __name__ == "__main__":

    if hp.training and hp.checking is False and hp.pre_processing is False:
        
        f = open(os.path.join(hp.train.model_path, 'info.txt'), 'w')
        for key in hp:
            if not isinstance(hp[key], dict):
                f.write(key + ': ' + str(hp[key]) + '\n')
            else:
                for subkey in hp[key]:
                    f.write(key + '.' + subkey + ': ' + str(hp[key][subkey]) + '\n')
        f.close()

        main()
        # train()
    elif hp.checking:
        check()


# def train():
#     writer = SummaryWriter(os.path.join(hp.train.model_path, hp.train.tensorboard_title))

#     train_dataset = my_dataset(hp.data.train_path, transform, hp.train.M)
#     test_dataset = my_dataset(hp.data.test_path, transform, hp.train.M)
#     train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers, drop_last=False)
#     test_loader = DataLoader(test_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers, drop_last=False)
#     plot_samples = construct_plot_samples(train_dataset, test_dataset)
#     image(plot_samples, writer, mode='oracle')
#     print(len(train_dataset))
#     print(len(test_dataset))

#     model = net(device)
#     toy = model.embedding_layer.projection.weight.clone()
#     model_dict = model.state_dict()
#     pretrained_dict = torch.load(hp.train.pretrained_path, map_location=torch.device('cpu'))
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     print(len(pretrained_dict))
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)
#     assert torch.sum(model.embedding_layer.projection.weight.data == toy) == 0
#     model.to(device)
#     for param in model.embedding_layer.parameters():
#         param.requires_grad = False
#     for name, param in model.named_parameters():
#         print(name, param.requires_grad)

#     class_loss = loss(coeff_cycle=hp.train.coeff_cycle)
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hp.train.lr)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hp.train.lr_decay_step, gamma=hp.train.lr_decay_coeff)
#     epoch_start = 0
#     if hp.train.resume:
#         checkpoint = torch.load(hp.train.resume_data_path, map_location=device)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         scheduler.load_state_dict(checkpoint['scheduler.state_dict'])
#         epoch_start = checkpoint['epoch']
    
#     for epoch in range(epoch_start, hp.train.epochs):
#         ret_train = train_module(train_loader, model, class_loss, optimizer, epoch)
#         ret_test = validation_module(test_loader, model, class_loss, epoch)
#         if optimizer.param_groups[0]['lr'] > hp.train.min_lr:
#             scheduler.step()

#         if (epoch + 1) % hp.train.plot_interval == 0:
#             image(plot_samples, writer, model=model, device=device, epoch=epoch, mode='test')

#         if (epoch + 1) % hp.train.checkpoint_interval == 0:
#             torch.save(model.state_dict(), os.path.join(hp.train.model_path, 'state_dict--epoch={}.pt').format(epoch + 1))
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'print_model': model.state_dict,
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict': scheduler.state_dict(),
#             }, os.path.join(hp.train.model_path, 'model.pt'))     

#         for label in ret_train:
#             writer.add_scalars(label, {'train': ret_train[label], 'test': ret_test[label]}, epoch)   
#     writer.close()
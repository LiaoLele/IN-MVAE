import torch
from dataset import my_dataset, my_dataset_test, tmp_dataset
# from data_generator.dataset import Dataset
# from data_generator.dataloader import DataLoader
# from data_generator.sampler import DataSampler

from config.hparam import hparam as hp
from torch.utils.data import DataLoader
from network import net
from utils import my_collate_fn
import time
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import fnmatch
# from utils import my_spectrogram
import librosa
import statistics as stat
import torchaudio
import torch.nn.functional as F


os.makedirs(hp.model.model_path, exist_ok=True)
device = torch.device(hp.device[0])
n_fft = int(hp.data.stft_frame * hp.data.sr)
hop = int(hp.data.stft_hop * hp.data.sr)
melsetting = {}
melsetting['n_fft'] = n_fft
melsetting['win_length'] = n_fft
melsetting['hop_length'] = hop
melsetting['n_mels'] = hp.model.feat_num
transform = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=hp.model.feat_num, melkwargs=melsetting)
    

def EER_calc(cosine_mat, batch_eer, file):
    diff = 1.0
    for thold in [0.01 * i + 0.1 for i in range(50)]:
        judge_positive = cosine_mat > thold
        frr = hp.test.N * hp.test.verify_num - torch.sum(torch.stack([judge_positive[i, :, i] for i in range(hp.test.N)]))
        far = judge_positive.sum() - torch.sum(torch.stack([judge_positive[i, :, i] for i in range(hp.test.N)]))
        frr = frr.float() / (hp.test.N * hp.test.verify_num)
        far = far.float() / (hp.test.N**2 * hp.test.verify_num - hp.test.N * hp.test.verify_num) 
        err = abs(far - frr)
        if err < diff:
            diff = err
            eer = (frr + far) / 2
            eer_thresh = thold
            eer_far = far
            eer_frr = frr
    batch_eer += eer
    # print('FILE: {4}  EER: {0:.2f}, FAR: {1:.2f}, FRR: {2:.2f}, threshold: {3:.2f}'.format(eer, eer_far, eer_frr, eer_thresh, file))
    return batch_eer


def cal_drop_p(epoch, point1, point2):

    if epoch < point1:
        return 0.0
    elif epoch >= point1 and epoch < point2:
        return 0.15 * (epoch - point1) / (point2 - point1)
    elif epoch >= point1 and epoch < 2000:
        return 0.15 * (2000 - epoch) / (2000 - point2)
    else: 
        return 0.0


def train_module(dataloader, model, optimizer, epoch):

    model.train()
    total_loss = 0.0

    for i, (data, label) in enumerate(dataloader):
        data = transform(data.float())
        data = data.float().cuda(device)
        label = label.long().cuda(device)
        with torch.no_grad():
            data = (data - data.mean(dim=-1, keepdim=True)) / (data.std(dim=-1, keepdim=True))
        # data = torch.rand(160, 40, 160).to(device)
        vec = model(data)

        loss = F.cross_entropy(vec, label, reduction='mean')

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        
        if (i + 1) % hp.train.log_interval == 0:
            mesg = "{0} Train Epoch:{1}[{2}/{3}], Loss:{4:.4f}, TLoss:{5:.4f}, drop_p:{6:.4f}".format(time.ctime(), epoch + 1, i + 1, len(dataloader),
                                                                                                      loss.item(), total_loss / (i + 1), model.drop_p)
            print(mesg) 
        i += 1  
    return total_loss / (i + 1)


def validation_module(dataloader, model, epoch, test_dataset):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader):
            data = transform(data.float())
            data = data.float().cuda(device)
            label = label.long().cuda(device)
            with torch.no_grad():
                data = (data - data.mean(dim=-1, keepdim=True)) / (data.std(dim=-1, keepdim=True))
            vec = model(data)

            loss = F.cross_entropy(vec, label, reduction='mean')
            total_loss += loss.item()
            i += 1
        mesg = "{0} Test Epoch:{1}[{2}/{3}], Loss:{4:.4f}, TLoss:{5:.4f}".format(time.ctime(), epoch + 1, i + 1, len(dataloader),
                                                                                    loss.item(), total_loss / (i + 1))
        print(mesg)
        test_dataset.reshuffle()
        return total_loss / (i + 1)


def train():
    total_epoch = 2000
    point1 = int(total_epoch * 0.2)
    point2 = int(total_epoch * 0.5)

    # writer = SummaryWriter(os.path.join(hp.model.model_path, hp.train.tensorboard_title))

    train_dataset = my_dataset(hp.data.train_path, hp.train.M, hp.data.train_start_time, hp.data.train_set, hp.data.train_spkrs, use_random_len=hp.train.use_random_len)
    test_dataset = tmp_dataset(hp.data.test_path, hp.train.M, hp.data.test_start_time, hp.data.test_spkrs, use_random_len=hp.train.use_random_len)
    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers,
                              drop_last=True, pin_memory=False, collate_fn=my_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers,
                             drop_last=True, pin_memory=False, collate_fn=my_collate_fn)

    # train_dataset = Dataset.from_pkl('/data/hdd0/zhaoyigu/DATASET/VoxCeleb/concatenate/dev/DataIdxSprd.pkl')
    # train_sampler = DataSampler(train_dataset).shuffle().batch(160)
    # train_loader = DataLoader(train_dataset, train_sampler, worker_num=4, collate_fn=my_collate_fn)

    print(len(train_loader))
    # print(len(test_loader))

    model = net(1211).cuda(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=hp.train.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1**(1 / 1000))
    if hp.train.resume:
        checkpoint = torch.load(hp.train.resume_data_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_start = checkpoint['epoch'] + 1
        train_loss_all = checkpoint['train_loss']
        test_loss_all = checkpoint["test_loss"]
        assert len(train_loss_all) == len(test_loss_all)
        # for i, train_loss in enumerate(train_loss_all):
        #     # writer.add_scalar('Loss', train_loss, i)
        #     writer.add_scalars('Loss', {'train_loss': train_loss, 'test_loss': test_loss_all[i]}, i)   
    else:
        epoch_start = 0
        train_loss_all, test_loss_all = [], []
    
    for epoch in range(epoch_start, hp.train.epochs):
        print(optimizer.param_groups[0]['lr'])

        if hp.train.use_dropout:
            drop_p = cal_drop_p(epoch, point1, point2)
            model.drop_p = drop_p

        tr_loss = train_module(train_loader, model, optimizer, epoch)
        # test_loss = validation_module(test_loader, model, epoch, test_dataset)
        train_loss_all.append(tr_loss)
        # test_loss_all.append(test_loss)

        # if epoch >= 1000 and optimizer.param_groups[0]['lr'] > hp.train.min_lr:
        #     scheduler.step()

        # if (epoch + 1) % hp.train.checkpoint_interval == 0:
        #     torch.save(model.state_dict(), os.path.join(hp.model.model_path, 'state_dict--epoch={}.pt'.format(epoch + 1)))

        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'print_model': model.state_dict,
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'scheduler_state_dict': scheduler.state_dict(),
        #     'train_loss': train_loss_all,
        #     'test_loss': test_loss_all,
        # }, os.path.join(hp.model.model_path, 'model.pt'))     

        # # writer.add_scalar('Loss', tr_loss, epoch)
        # writer.add_scalars('Loss', {'train_loss': tr_loss, 'test_loss': test_loss}, epoch)   
    writer.close()


def verification():

    test_dataset = my_dataset_test(hp.test.test_path, hp.test.enroll_num, hp.test.verify_num, 
                                   frame_num=hp.test.frame_num, enroll_hop=hp.test.enroll_hop,
                                   verify_hop=hp.test.verify_hop, num_speaker=hp.test.num_speaker)
    test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)
    model = speaker_net().to(device)
    model_file = list(filter(lambda x: fnmatch.fnmatch(x, 'state_dict*=*.pt'), os.listdir(hp.test.model_path)))
    fb_mat = torch.from_numpy(librosa.filters.mel(hp.data.sr, n_fft, n_mels=hp.data.nmels)).unsqueeze(0)

    for file in model_file:
        state_dict = torch.load(os.path.join(hp.test.model_path, file), map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        total_eer = 0.0
        # total_acc_rate = 0.0
        with torch.no_grad():
            for epoch in range(hp.test.epochs):
                batch_eer = 0.0
                # batch_acc_rate = []
                for i, data in enumerate(test_loader):
                    data = data.contiguous().view(-1, data.shape[-1]).to(device).float()
                    data = my_spectrogram(data, n_fft, hop)
                    data = torch.matmul(fb_mat.cuda(device), data)
                    data = 10 * torch.log10(torch.clamp(data, 1e-10))
                    data = data.view(hp.test.N, -1, data.shape[-2], data.shape[-1])
                    enrollment, verification = data[:, 0:hp.test.enroll_num, :, :], data[:, hp.test.enroll_num:, :, :]  # enroll [batch, M/2, time, n_feature]; valid [batch, M/2, time, n_feature]
                    enrollment = enrollment.contiguous().view(-1, enrollment.shape[-2], enrollment.shape[-1]).to(device).float()
                    verification = verification.contiguous().view(-1, verification.shape[-2], verification.shape[-1]).to(device).float()

                    enrollment_embeddings = model.embedding_layer(enrollment.permute(0, 2, 1))
                    verification_embeddings = model.embedding_layer(verification.permute(0, 2, 1))
                    enrollment_embeddings = enrollment_embeddings.view(hp.test.N, hp.test.enroll_num, -1)
                    verification_embeddings = verification_embeddings.view(hp.test.N, hp.test.verify_num, -1)

                    centroids = torch.mean(enrollment_embeddings, dim=1)  # [N, d_embedding]
                    cosine_mat = get_cosmat(verification_embeddings, centroids, mode='inclusive')  # [N, verify_num, N]

                    """ SOFTMAX """
            #         softmax_mat = F.softmax(cosine_mat, dim=2)
            #         _, label = torch.max(softmax_mat, dim=2, keepdim=True)
            #         acc_rate = torch.sum(torch.stack([label[i, :, 0] == i for i in range(hp.test.N)])).float() / (hp.test.N * hp.test.verify_num)
            #         batch_acc_rate.append(acc_rate.item())
            #     print('epoch {} in file {}'.format(epoch + 1, file))
            #     # total_acc_rate += batch_acc_rate / (i + 1)
            #     # print("{} Softmax accurate rate is {:.4f} in epoch {}".format(file, batch_acc_rate / (i + 1), epoch + 1))
            # total_acc_rate = stat.mean(batch_acc_rate)
            # total_acc_std = stat.stdev(batch_acc_rate)
            # print("\n{} Total softmax accurate rate is {:.4f}\nTotal softmax accurate std is {:.4f}\n".format(file, total_acc_rate, total_acc_std))
            # with open(os.path.join(hp.test.model_path, 'Softmax Accuracy result for test set.txt'), 'a+') as f:
            #     f.write("FILE: {2}---Total softmax accurate and std across {0} epochs: {1:.4f}\t{3:.4f}\n".format(hp.test.epochs, total_acc_rate, file, total_acc_std))
            # f.close()
            # if total_acc_rate > best_performance:
            #     best_performance = total_acc_rate
            #     torch.save(checkpoint['model_state_dict'], os.path.join(hp.test.model_path, 'best_state_dict--softmax.pt'))

                    """ ERR """
                    batch_eer = EER_calc(cosine_mat, batch_eer, file)
                total_eer += batch_eer / (i + 1)
            total_eer = total_eer / hp.test.epochs
            print("\n FILE: {2}---EER across {0} epochs: {1:.4f}\n".format(hp.test.epochs, total_eer, file))
            with open(os.path.join(hp.test.model_path, 'EER_result.txt'), 'w') as f:
                f.write("FILE: {2}---EER across {0} epochs: {1:.4f}".format(hp.test.epochs, total_eer, file))
            f.close()
            # if total_eer < best_performance:
            #     best_performance = total_acc_rate
            #     torch.save(checkpoint['model_state_dict'], os.path.join(hp.test.model_path, 'best_state_dict--softmax.pt'))


def correction():
    writer = SummaryWriter(os.path.join(hp.model.model_path, hp.train.tensorboard_title))
    checkpoint = torch.load(hp.train.resume_data_path, map_location=device)
    train_dataset = my_dataset(hp.data.train_path, hp.train.M, hp.data.train_start_time, hp.data.train_set, hp.data.train_spkrs, use_random_len=hp.train.use_random_len)
    test_dataset = tmp_dataset(hp.data.test_path, hp.train.M, hp.data.test_start_time, hp.data.test_set, hp.data.test_spkrs, use_random_len=hp.train.use_random_len)
    test_loader = DataLoader(test_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers,
                             drop_last=True, pin_memory=False, collate_fn=my_collate_fn)
    test_loss_correct = []
    res = [5]

    model = net(len(train_dataset)).cuda(device)
    model_file = list(filter(lambda x: x.startswith('state'), os.listdir('/data/hdd0/zhaoyigu/PROJECT/Xvector_speaker_encoder_data/test_voxceleb_03/model/')))
    model_file = sorted(model_file, key=lambda x: int(x.split('=')[1].split('.')[0]))
    for i, state_dict in enumerate(model_file):
        model.load_state_dict(torch.load(os.path.join('/data/hdd0/zhaoyigu/PROJECT/Xvector_speaker_encoder_data/test_voxceleb_03/model/', state_dict), 
                                         map_location=device))
        test_loss = validation_module(test_loader, model, 0)
        test_loss_correct.extend(list(np.linspace(res[-1], test_loss, num=99 if i == 0 else 100, endpoint=False)))
        res.append(test_loss)

    model.load_state_dict(checkpoint['model_state_dict'])
    checkpoint['test_loss_correct'] = test_loss_correct
    test_loss = validation_module(test_loader, model, 0)
    test_loss_correct.extend(list(np.linspace(res[-1], test_loss, num=checkpoint['epoch'] - 1699 + 1, endpoint=True)))
    res.append(test_loss)
    assert len(test_loss_correct) == len(checkpoint['train_loss'])

    torch.save({
        'epoch': checkpoint['epoch'],
        'model_state_dict': checkpoint['model_state_dict'],
        'print_model': checkpoint['print_model'],
        'optimizer_state_dict': checkpoint['optimizer_state_dict'],
        'scheduler_state_dict': checkpoint['scheduler_state_dict'],
        'train_loss': checkpoint['train_loss'],
        'test_loss': checkpoint['test_loss'],
        'test_loss_correct': test_loss_correct,
    }, os.path.join(hp.model.model_path, 'model_new.pt'))  

    

    

        


if __name__ == "__main__":
    def plot():
        root = '/data/hdd0/zhaoyigu/PROJECT/GE2E_speaker_encoder_data/test_02/model/'
        legend = ['training data', 'test data']
        txt_file = [
            root + 'Softmax Accuracy result for training set.txt',
            root + 'Softmax Accuracy result for test set.txt',
        ]
        total_acc = []
        for txt in txt_file:
            acc = []
            f = open(txt, 'r')
            while True:
                line = f.readline()
                if not line:
                    break
                epoch = int(line.rsplit('=')[1].rsplit('.')[0]) + 1
                acc.append((int(line[-5: -1]) * 1e-2, epoch))
            acc = sorted(acc, key=lambda x: x[1])
            acc, x_ticks = [x for x, _ in acc], [str(y) for _, y in acc]
            acc = np.array(acc)
            total_acc.append(acc)
        total_acc = np.stack(total_acc, 0)
        fig, ax = plt.subplots()
        ax.set_title('Accuracy rate')
        for i, acc in enumerate(total_acc):
            plt.plot(np.arange(1, acc.shape[0] + 1), acc, label=legend[i])

        ax.set_xlabel('epoch')
        ax.set_xticks(np.arange(1, 11))
        ax.set_xticklabels(x_ticks)
        ax.set_yticks(np.arange(95, 100.5, 0.5))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.set_ylabel('Accuracy rate [%]')
        ax.legend()
        plt.show()
        fig.savefig(os.path.join(root, 'accuracy rate.png'))

    if hp.training:
        if hp.train.resume is False:
            f = open(os.path.join(hp.model.model_path, 'net.py'), 'w')
            f2 = open('network.py', 'r')
            thisnet = f2.read()
            f2.close()
            f.write(thisnet)
            f.close()

            f = open(os.path.join(hp.model.model_path, 'info.txt'), 'w')
            for key in hp:
                if not isinstance(hp[key], dict):
                    f.write(key + ': ' + str(hp[key]) + '\n')
                else:
                    for subkey in hp[key]:
                        f.write(key + '.' + subkey + ': ' + str(hp[key][subkey]) + '\n')
            f.close()
        train()
    else:
        verification()
        # plot()
    # correction()

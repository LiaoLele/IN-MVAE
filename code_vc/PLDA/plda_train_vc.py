import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.realpath(os.path.dirname(__file__)))))
import torch
import torchaudio
import pickle
import numpy as np
import bob.learn.em
import bob.io.base
import bob.learn.linear
from collections import defaultdict
from torch.utils.data import DataLoader
from utils import CollateFnWrapper
from hparam_pldatrain import hparam as hp
from dataset_plda import Dataset
from dataset_plda import TrainBatchSampler
from network_cnn import AutoEncoder
from config import cfgs
from PreProcess.data_utils import Spectrogram


def get_embds(egsfile_path, out_path, device, use_slide=False, data_prefix=None, suffix='0', sub_state=0):
    """ Compute xvectors of training set and calculate the mean of all xvectors """
    """ 
    Args:
        `egsfile_path`: path where training egs.pkl is saved
        `model_path`: path where speaker encoder network is saved
        `out_path`: path where xvectors will be saved
        `device`: torch device
        `prefix`: path prefix
    
    Out:
        `xvec_all`: [Not returned but saved][dict]
                    {spkr-id-0: np.ndarray of shape (nsample, embedding_dim), spkr-id-1: ..., ...}
        `xvec_mean`: [Not returned but saved][np.ndarray] mean vector of all xvectors
    """
    os.makedirs(out_path, exist_ok=True)

    if sub_state == 0:
        """ Extract Xvectors """
        if os.path.exists(os.path.join(out_path, "xvec_list.train." + suffix + ".pkl")):
            with open(os.path.join(out_path, "xvec_list.train." + suffix + ".pkl"), 'rb') as f:
                xvec_all = pickle.load(f)
        else:
            with open(os.path.join(out_path, 'xvec_info.train.' + suffix + '.txt'), 'w') as f:
                print('pldatraindata: {}'.format(egsfile_path), file=f)
                # print("xvector_model: {}".format(model_path), file=f)
            raw_dataset = Dataset.from_pkl(egsfile_path, data_prefix)
            sampler = TrainBatchSampler.from_dataset(raw_dataset, 16, drop_last=False, n_batch=None)
            dataloader = DataLoader(raw_dataset, batch_sampler=sampler, num_workers=0, pin_memory=False, collate_fn=CollateFnWrapper)
            print("PLDA Train_loader length: {}".format(int(len(dataloader))))

            state_dict = torch.load(os.path.join(cfgs.general.prefix, cfgs.path.model), map_location=device)
            # state_dict = torch.load(f'{os.path.join(config.general.prefix, config.path.model)}', map_location='cpu')
            model = AutoEncoder(cfgs)
            model.load_state_dict(state_dict)
            model.to(cfgs.general.device)
            model.eval()

            transform = Spectrogram(**cfgs.sigproc)

            xvec_all = defaultdict(list)
            for i, in_data in enumerate(dataloader): 
                if (i + 1) % 30 == 0:
                    print("Processing {}/{}".format(i + 1, len(dataloader)))
                with torch.no_grad():
                    data, label = in_data.data, in_data.label
                    label = label.numpy()
                    # data = transform(data.float())
                    data = data.to(device).float()
                    data = (data - data.mean(dim=-1, keepdim=True)) / (data.std(dim=-1, keepdim=True))
                    _, data, _ = transform(data)
                    vec = model.get_speaker_embeddings(data)
                    vec = vec.cpu().numpy().astype(np.float64)

                    for i, spkr_idx in enumerate(label):
                        xvec_all[spkr_idx].append(vec[i, :])
            with open(os.path.join(out_path, "xvec_list.train." + suffix + ".pkl"), 'wb') as f:
                pickle.dump(xvec_all, f)

    if sub_state <= 1:
        """ Stack xvec_list into ndarray """
        """ Calculate mean """
        if sub_state == 1:
            with open(os.path.join(out_path, "xvec_list.train." + suffix + ".pkl"), 'rb') as f:
                xvec_all = pickle.load(f)
        # """ Stack xvec_list into ndarray """
        num_egs_all = []
        xvec_mean_all = []
        xvec_mean = 0
        xvec_all_keys = list(xvec_all.keys())
        for spkr_idx in xvec_all_keys:
            print("calculate mean for speaker {}".format(spkr_idx.item()))
            xvec_all[spkr_idx] = np.stack(xvec_all.pop(spkr_idx), axis=0)
            num_egs_all.append(xvec_all[spkr_idx].shape[0])
            xvec_mean_all.append(np.mean(xvec_all[spkr_idx], axis=0, keepdims=True))
        with open(os.path.join(out_path, 'xvec_ndarray.train.' + suffix + '.pkl'), 'wb') as f:
            pickle.dump(xvec_all, f) 

        # """ Calculate mean """
        for i, mean in enumerate(xvec_mean_all):
            xvec_mean = xvec_mean + mean * num_egs_all[i] / sum(num_egs_all)
        if xvec_mean.ndim != 2:
            import ipdb; ipdb.set_trace()
        with open(os.path.join(out_path, 'xvec_mean.train.' + suffix + '.pkl'), 'wb') as f:
            pickle.dump(xvec_mean, f)

    if sub_state <= 2:
        """ Subtract mean """ 
        if sub_state == 2:
            with open(os.path.join(out_path, 'xvec_ndarray.train.' + suffix + '.pkl'), 'rb') as f:
                xvec_all = pickle.load(f)
            with open(os.path.join(out_path, 'xvec_mean.train.' + suffix + '.pkl'), 'rb') as f:
                xvec_mean = pickle.load(f)
        xvec_all_keys = list(xvec_all.keys())
        for spkr_id in xvec_all_keys:
            xvec_all[spkr_id] = xvec_all.pop(spkr_id) - xvec_mean
        with open(os.path.join(out_path, 'xvec_ndarray.zero_mean.train.' + suffix + '.pkl'), 'wb') as f:
            pickle.dump(xvec_all, f)

    if sub_state <= 3:
        """ check mean """
        if sub_state == 3:
            with open(os.path.join(out_path, 'xvec_ndarray.zero_mean.train.' + suffix + '.pkl'), 'rb') as f:
                xvec_all = pickle.load(f)
        num_egs_all = []
        xvec_mean_all = []
        xvec_mean = 0
        xvec_all_keys = list(xvec_all.keys())
        for spkr_idx in xvec_all_keys:
            num_egs_all.append(xvec_all[spkr_idx].shape[0])
            xvec_mean_all.append(np.mean(xvec_all[spkr_idx], axis=0, keepdims=True))
        for i, mean in enumerate(xvec_mean_all):
            xvec_mean = xvec_mean + mean * num_egs_all[i] / sum(num_egs_all)
        # import ipdb; ipdb.set_trace()
    

def get_lda_transform(xvec_path, out_path, lda_dim=128, centerization=True, suffix='0'):
    """ Get LDA tranformation matrix """
    """ 
    Args:
        `xvec_path`: path where training xvectors are saved
        `out_path`: path where lda machine will be saved
        `lda_dim`: dimension of lda
        `centerization`: True to subtract mean
    
    Out:
        `lda_machine`: [Not returned but saved][bob.learn.linear.Machine]
        `eigen_values`: [Not returned but saved][np.ndarray] eigenvalues of Sw-1Sb
    """
    os.makedirs(out_path, exist_ok=True)
    f_txt = open(os.path.join(out_path, 'lda.info.' + suffix + '.txt'), 'w')
    print('xvectorpath: {}'.format(xvec_path), file=f_txt)
    print('Centralization: {}'.format(centerization), file=f_txt)
    print('lda_dim: {}'.format(lda_dim), file=f_txt)
    with open(xvec_path, 'rb') as f:
        xvec_all = pickle.load(f)
    print("Speaker number: {}".format(len(xvec_all.keys())), file=f_txt)
    print("Speaker number is {}".format(len(xvec_all.keys())))
    f_txt.close()

    """ LDA parameter estimation """
    machine_file = bob.io.base.HDF5File(os.path.join(out_path, 'lda_machine.withdimreduction.' + suffix + '.hdf5'), 'w')
    machine_file_whole = bob.io.base.HDF5File(os.path.join(out_path, 'lda_machine.withoutdimreduction.' + suffix + '.hdf5'), 'w')
    trainer = bob.learn.linear.FisherLDATrainer()
    trainer.strip_to_rank = False
    print("start lda training!")
    lda_machine, eigen_values = trainer.train(xvec_all.values())
    lda_machine.save(machine_file_whole)
    print("Save the Lda machine")
    lda_machine.resize(lda_machine.shape[0], lda_dim)
    lda_machine.save(machine_file)
    print("Save the dim reduced lda machine")
    with open(os.path.join(out_path, 'lda_machine_eigen_value.' + suffix + '.pkl'), 'wb') as f:
        pickle.dump(eigen_values, f)


def get_plda_machine(lda_path, xvec_path, out_path, iter_num=15, centerization=True, length_normalization=True, suffix='0', sub_state=0):
    """ Get pldabase and dimension reduced lda """
    """ 
    Args:
        `lda_path`: path where lda_machine is saved
        `xvec_path`: path where xvectors are saved. 
        `out_path`: path where PLDAbase will be saved
        `iter_num`: number of iterations in em algorithm
        `centerization`: whether to subtract mean from xvectors
        `length_normalization`: whether to conduct normalization in plda training

    Out:
        xvec_train_subtract-mean-lda.pkl: [dict] file where xvectors after mean normalization and dimension reduction are saved
        whitening_matrix.pkl: [np.ndarray] file where tranformation matrix for whitening is saved
        plda_base: [Not returned but saved][bob.learn.em.PLDABase]
    """
    os.makedirs(out_path, exist_ok=True)

    if sub_state == 0:
        """ Calculate xvector after dim reduction and possible length normalization """
        with open(os.path.join(out_path, 'xvec_afterlda.info.' + suffix + '.txt'), 'w') as f:
            print('xvector_path: {}'.format(xvec_path), file=f)
            print("lda_path: {}".format(lda_path), file=f)
            print("Centralization: {}".format(centerization), file=f)
        with open(xvec_path, 'rb') as f:
            xvec_all = pickle.load(f)
        machine_file = bob.io.base.HDF5File(lda_path)
        lda_machine = bob.learn.linear.Machine(machine_file)
        del machine_file

        print("Transform through LDA!")
        xvec_all_keys = sorted(list(xvec_all.keys()))
        for spkr_id in xvec_all_keys:
            print("Transforming {}th speaker".format(spkr_id))
            xvec_all[spkr_id] = lda_machine.forward(xvec_all.pop(spkr_id))
        with open(os.path.join(out_path, 'xvec_afterlda.withoutlengthnorm.' + suffix + '.pkl'), 'wb') as f:
            pickle.dump(xvec_all, f)

        print("Conduct length normalization")
        for spkr_id in xvec_all_keys:
            xvec_all[spkr_id] = xvec_all[spkr_id] / np.linalg.norm(xvec_all[spkr_id], axis=1, keepdims=True)
        with open(os.path.join(out_path, 'xvec_afterlda.withlengthnorm.' + suffix + '.pkl'), 'wb') as f:
            pickle.dump(xvec_all, f)

    if sub_state <= 1:
        """ Train PLDA model """
        if length_normalization:
            xvec_lda_path = os.path.join(out_path, 'xvec_afterlda.withlengthnorm.' + suffix + '.pkl')
        else:
            xvec_lda_path = os.path.join(out_path, 'xvec_afterlda.withoutlengthnorm.' + suffix + '.pkl')
        with open(os.path.join(out_path, 'plda.info.' + suffix + '.txt'), 'w') as f:
            print("xvector_after_lda_path: {}".format(xvec_lda_path), file=f)
            print("length normalization: {}".format(length_normalization), file=f)
            print("Centralization: {}".format(centerization), file=f)
        with open(xvec_lda_path, 'rb') as f:
            xvec_all = pickle.load(f)

        # dict to list
        xvec_all_list = []
        xvec_all_keys = list(xvec_all.keys())
        for spkr_id in xvec_all_keys:
            xvec_all_list.append(xvec_all.pop(spkr_id))
    
        # xvec = np.concatenate(xvec_all_list, axis=0)
        # import ipdb; ipdb.set_trace()
        
        # if sub_state < 2:
        #     print("PCA Whitening and length normalization")
        #     cov_mat = np.matmul(np.concatenate(xvec_all_list, axis=0).transpose(), np.concatenate(xvec_all_list, axis=0))
        #     U, S, V = np.linalg.svd(cov_mat, hermitian=True)
        #     whitening_matrix = np.diag(1 / (np.sqrt(S) + 1e-5))@V
        #     with open(os.path.join(os.path.dirname(lda_path), 'whitening_matrix.pkl'), 'wb') as f:
        #         pickle.dump(whitening_matrix, f)
        # elif sub_state == 2:
        #     print("Read whitening matrix")
        #     with open(os.path.join(os.path.dirname(lda_path), 'whitening_matrix.pkl'), 'rb') as f:
        #         whitening_matrix = pickle.load(f)

        # PLDA
        plda_trainer = bob.learn.em.PLDATrainer()
        plda_trainer.use_sum_second_order = True
        plda_trainer.init_f_method = "BETWEEN_SCATTER"
        plda_trainer.init_g_method = "WITHIN_SCATTER"
        plda_trainer.init_sigma_method = "VARIANCE_DATA"
        variance_flooring = 1e-5
        feature_dim = xvec_all_list[0].shape[1]
        plda_base = bob.learn.em.PLDABase(feature_dim, feature_dim, feature_dim - 1, variance_flooring)
        print("Start PLDA training!")
        bob.learn.em.train(plda_trainer, plda_base, xvec_all_list, max_iterations=iter_num)
        plda_hdf5 = bob.io.base.HDF5File(os.path.join(out_path, "plda_base." + suffix + ".hdf5"), 'w')
        print("save plda base to file")
        plda_base.save(plda_hdf5)
        del plda_hdf5
  

def main(state=0, sub_state=0, data_prefix=None):
    if state == 0:
        """ extract embeddings from training data
        and calculate mean of all embeddings """
        traindata_path = hp.path.traindata_path
        out_path = '/home/nis/lele.liao/projects/code_vc/PLDA/data_vc'
        device = torch.device(hp.device)
        # suffix = 'augdata'
        get_embds(traindata_path, out_path, device, use_slide=False,
                  data_prefix=data_prefix, sub_state=sub_state)

    elif state == 1:
        """ get LDA transformation matrix """
        xvec_path = os.path.join('/home/nis/lele.liao/projects/code_vc/PLDA/data_vc', 'xvec_ndarray.zero_mean.train.0.pkl')
        # xvec_path = os.path.join('/home/nis/lele.liao/projects/code_vc/PLDA/data_vc', 'xvec_ndarray.train.0.pkl')
        out_path = '/home/nis/lele.liao/projects/code_vc/PLDA/data_vc'
        lda_dim = 64
        # suffix = '0'
        get_lda_transform(xvec_path, out_path, lda_dim=lda_dim, centerization=True, suffix='0')

    elif state == 2:
        """ get PLDAbase and whitening matrix """
        # suffix = '0'
        lda_path = os.path.join('/home/nis/lele.liao/projects/code_vc/PLDA/data_vc', 'lda_machine.withdimreduction.0.hdf5')
        xvec_path = os.path.join('/home/nis/lele.liao/projects/code_vc/PLDA/data_vc', 'xvec_ndarray.zero_mean.train.0.pkl')
        # xvec_path = os.path.join('/home/nis/lele.liao/projects/code_vc/PLDA/data_vc', 'xvec_ndarray.train.0.pkl')
        out_path = '/home/nis/lele.liao/projects/code_vc/PLDA/data_vc'
        iter_num = 15
        get_plda_machine(lda_path, xvec_path, out_path, iter_num=iter_num, centerization=True, length_normalization=True, sub_state=sub_state)


if __name__ == "__main__":
    main(state=2, sub_state=0, data_prefix=hp.path.data_prefix)
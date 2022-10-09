import numpy as np
import time


def avgmm(mix, n_iter, state_num=2, convg_tol=1e-6):

    n_freq, n_src, n_frame = mix.shape  # mix [F, N, T]
    eps = 1e-10
    flag = False

    """ whitening """
    Cov_mat = mix @ (mix.conj().transpose(0, 2, 1))
    # Cov_mat = np.zeros((n_src, n_src), dtype=mix.dtype)
    # for i in range(mix.shape[2]):
    #     if i == 113:
    #         import ipdb;ipdb.set_trace()
    #     Cov_mat = Cov_mat + mix[0, :, None, i] @ mix[0, None, :, i].conj()
    #     print("{}: {}".format(i, np.linalg.matrix_rank(Cov_mat))) 
    for k in range(mix.shape[0]):
        if np.linalg.matrix_rank(Cov_mat[k, :, :]) < 2:
            Cov_mat[k, :, :] = Cov_mat[k, :, :] + 0.001 * np.min(np.abs(Cov_mat[k, :, :])) * np.eye(n_src)
    s, U = np.linalg.eig(Cov_mat)
    S = np.stack([np.diag(ev**(-0.5)) for ev in s])
    whitening_mat = U @ S @ (U.conj().transpose(0, 2, 1))  # [F, N, N]
    mix = whitening_mat @ mix

    """ Initialization """
    sep_mat = np.stack([np.eye(n_src, dtype=mix.dtype) for _ in range(n_freq)])    # sep_mat [F, N, N]
    sep = sep_mat @ mix                                                            # sep [F, N, T]
    sep_pow = np.power(np.abs(sep), 2).transpose(1, 0, 2)                          # sep_pow [N, F, T]
    np.clip(sep_pow, a_min=eps, a_max=None, out=sep_pow)
    h = np.random.uniform(low=0.999, high=1.001, size=(n_src, n_frame))            # h [N, T]
    nu = np.random.uniform(low=0.999, high=1.001, size=(state_num, n_freq, n_src))  # nu [state_num, F, N]
    prior_p = 1 / state_num**2 * np.ones((state_num, state_num))                   # prior_p [state_num, state_num]
    Q_st = 1 / state_num**2 * np.ones((state_num, state_num, n_frame))                               # Q_st [state_num, state_num, T]
    Q_1 = np.sum(Q_st, axis=1)  # [state_num, n_frame]
    Q_2 = np.sum(Q_st, axis=0)  # [state_num, n_frame]

    """ EM algorithm """
    pObj = np.inf
    end = 0
    for t in range(n_iter):
        start = time.time()
        """ Update h """
        h[0, :] = n_freq / (eps + np.sum(Q_1[:, None, :] * nu[:, :, 0, None] * sep_pow[0, None, :, :], axis=(0, 1)))
        h[1, :] = n_freq / (eps + np.sum(Q_2[:, None, :] * nu[:, :, 1, None] * sep_pow[1, None, :, :], axis=(0, 1)))

        pophi = np.expand_dims(nu.transpose(2, 0, 1), 3) * h[:, None, None, :]         # pophi [N, state_num, F, T]
        pophi_y = pophi * sep_pow[:, None, :, :]

        """ Updata Q_st """
        logGaussProb = np.sum(np.log(pophi + eps) - pophi_y, axis=2)   # [N, D, T]
        logGaussProb = logGaussProb[0, :, None, :] + logGaussProb[1, None, :, :]
        logGaussProb_copy = logGaussProb.copy()
        mark = np.where(prior_p == 0)
        if np.size(mark[0]) != 0:
            logGaussProb[mark] = np.min(logGaussProb, axis=(0, 1))
        logGaussProb = logGaussProb - np.max(logGaussProb, axis=(0, 1), keepdims=True)
        GaussProb = np.exp(logGaussProb) * prior_p[:, :, None]
        Q_st = GaussProb / np.sum(GaussProb, axis=(0, 1))

        Q_1 = np.sum(Q_st, axis=1)  # [state_num, n_frame]
        Q_2 = np.sum(Q_st, axis=0)  # [state_num, n_frame]
        log_pg = np.clip(np.log(prior_p[:, :, None]), a_min=-200, a_max=None) + logGaussProb_copy

        """ Updata sep_mat """
        phi = pophi[0, :, None, :, :] - pophi[1, None, :, :, :]   # [D,D,F,T]
        M = np.sum(phi * Q_st[:, :, None, :], axis=(0, 1))  # [F, T]
        M = (mix * M[:, None, :]) @ (mix.conj().transpose(0, 2, 1))  # [F, N, N]
        beta = np.real(M[:, 0, 0] + M[:, 1, 1]) / 2 - np.sqrt((np.real(M[:, 0, 0] - M[:, 1, 1]) / 2)**2 + np.abs(M[:, 0, 1])**2)
        a_star = 1 / np.sqrt(1 + np.abs((beta - np.real(M[:, 0, 0])) / M[:, 0, 1])**2)
        b_star = ((beta - np.real(M[:, 0, 0])) / M[:, 0, 1]) * a_star
        sep_mat[:, 0, 0] = a_star.conj()
        sep_mat[:, 1, 1] = a_star
        sep_mat[:, 0, 1] = b_star.conj()
        sep_mat[:, 1, 0] = -b_star

        sep = sep_mat @ mix                                                            # sep [F, N, T]
        sep_pow = np.power(np.abs(sep), 2).transpose(1, 0, 2)                          # sep_pow [N, F, T]
        np.clip(sep_pow, a_min=eps, a_max=None, out=sep_pow)

        """ Updata nu [state_num, F, N]"""
        nu[:, :, 0] = np.sum(Q_1, axis=1)[:, None] / (eps + np.sum(Q_1[:, None, :] * h[0, None, None, :] * sep_pow[0, None, :, :], axis=2)) 
        nu[:, :, 1] = np.sum(Q_2, axis=1)[:, None] / (eps + np.sum(Q_2[:, None, :] * h[1, None, None, :] * sep_pow[1, None, :, :], axis=2))

        """ Updata ps """
        prior_p = np.mean(Q_st, axis=2)

        end = end + time.time() - start
        Obj = np.sum(Q_st * log_pg)
        if np.isnan(Obj):
            flag = True
        dObj = abs(Obj - pObj) / abs(Obj)
        pObj = Obj
        if dObj < convg_tol:
            break

        """ # display infomation """
        if t % 10 == 0:
            print('avgmm_Iterations: {}, Objective: {}, dObj: {}'.format(t, Obj, dObj))

    sep_mat_fin = sep_mat @ whitening_mat
    for k in range(n_freq):
        sep_mat_fin[k, :, :] = \
            np.diag(np.diag(np.linalg.inv(sep_mat_fin[k, :, :]))) @ sep_mat[k, :, :]
    sep = sep_mat_fin @ mix

    return sep, flag, end, end / t



if __name__ == "__main__":

    import fnmatch
    import os
    import librosa as rosa
    import mir_eval.separation as bss_eval
    # import pandas as pd
    import soundfile as sf

    # df = pd.DataFrame(columns=['Filename', 'SDR_ori', 'SDR', 'SIR_ori', 'SIR', 'perm'])

    # mix_wav_path = '/home/user/zhaoyi.gu/mnt/g2/PROJECT/MVAE_w_embds_data/mixture/vctk_90_noshuffle/p273-0_p284-0_snr-0_t60-0.15_mix.wav'
    # src_wav_path = '/home/user/zhaoyi.gu/mnt/g2/PROJECT/MVAE_w_embds_data/mixture/vctk_90_noshuffle/p273-0_p284-0_src.wav'
    # mix_wav_path = '/home/user/zhaoyi.gu/mnt/g2/PROJECT/MVAE_w_embds_data/mixture/vctk_target-11_noise-11_noshuffle/p241-0_p234-0_snr-0_t60-0.15_mix.wav'
    # src_wav_path = '/home/user/zhaoyi.gu/mnt/g2/PROJECT/MVAE_w_embds_data/mixture/vctk_target-11_noise-11_noshuffle/p241-0_p234-0_src.wav'
    # output_path = '/home/user/zhaoyi.gu/mnt/g2/PROJECT/MVAE_w_embds_data/output/ilrma_ab_div/'
    mix_wav_path = '/data/hdd0/zhaoyigu/DATASET/tmp/male_female_aec.wav'
    output_path = '/data/hdd0/zhaoyigu/DATASET/tmp/ilrma/sep_avgmm_4ch.wav'

    audio, _ = rosa.load(mix_wav_path, sr=16000, mono=False)
    # audio = audio[0:2,:]
    # src, _ = rosa.load(src_wav_path, sr=16000, mono=False)
    mix_spec = np.stack([rosa.stft(np.asfortranarray(x_c), n_fft=1024, hop_length=256) for x_c in audio], axis=1)
    sep_spec, flag, time_all, avg_time = avgmm(mix_spec, 1000, state_num=2)
    sep = np.stack([rosa.istft(sep_spec[:, ch, :], hop_length=256) for ch in range(0, 4)], axis=0)
    sf.write(output_path, sep.T, 16000)
    print(time_all)
    print(avg_time)






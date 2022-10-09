import numpy as np
import soundfile as sf

EPS = 3e-16


def ilrma(mix, n_iter, n_basis=2, convg_tol=1e-6):
    """Implementation of ILRMA (Independent Low-Rank Matrix Analysis).
    This algorithm is called ILRMA1 in http://d-kitamura.net/pdf/misc/AlgorithmsForIndependentLowRankMatrixAnalysis.pdf
    It only works in determined case (n_sources == n_channels).

    Args:
        mix (numpy.ndarray): (n_frequencies, n_channels, n_frames)
            STFT representation of the observed signal.
        n_iter (int): Number of iterations.
        n_basis (int): Number of basis in the NMF model.
        proj_back (bool): If use back-projection technique.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Tuple of separated signal and
            separation matrix. The shapes of separated signal and separation
            matrix are (n_frequencies, n_sources, n_frames) and
            (n_frequencies, n_sources, n_channels), respectively.
    """
    flag = False
    n_freq, n_src, n_frame = mix.shape

    sep_mat = np.stack([np.eye(n_src, dtype=mix.dtype) for _ in range(n_freq)])
    basis = np.abs(np.random.randn(n_src, n_freq, n_basis))
    act = np.abs(np.random.randn(n_src, n_basis, n_frame))
    # basis = np.random.uniform(low=0.999, high=1.001, size=(n_src, n_freq, n_basis))
    # act = np.random.uniform(low=0.999, high=1.001, size=(n_src, n_basis, n_frame))
    sep = sep_mat @ mix
    sep_pow = np.power(np.abs(sep), 2)  # (n_freq, n_src, n_frame)
    model = basis @ act  # (n_src, n_freq, n_frame)
    m_reci = 1 / model

    eye = np.tile(np.eye(n_src), (n_freq, 1, 1))

    pObj = np.inf
    for t in range(n_iter):
        for src in range(n_src):
            h = (sep_pow[:, src, :] * m_reci[src]**2) @ act[src].T
            h /= m_reci[src] @ act[src].T
            h = np.sqrt(h, out=h)
            basis[src] *= h
            np.clip(basis[src], a_min=EPS, a_max=None, out=basis[src])

            model[src] = basis[src] @ act[src]
            m_reci[src] = 1 / model[src]

            h = basis[src].T @ (sep_pow[:, src, :] * m_reci[src]**2)
            h /= basis[src].T @ m_reci[src]
            h = np.sqrt(h, out=h)
            act[src] *= h
            np.clip(act[src], a_min=EPS, a_max=None, out=act[src])

            model[src] = basis[src] @ act[src]
            m_reci[src] = 1 / model[src]

            h = m_reci[src, :, :, None] @ np.ones((1, n_src))
            h = mix.conj() @ (mix.swapaxes(1, 2) * h)
            u_mat = h.swapaxes(1, 2) / n_frame
            h = sep_mat @ u_mat + EPS * eye
            sep_mat[:, src, :] = np.linalg.solve(h, eye[:, :, src]).conj()
            h = sep_mat[:, src, None, :] @ u_mat
            h = (h @ sep_mat[:, src, :, None].conj()).squeeze(2)
            sep_mat[:, src, :] = (sep_mat[:, src, :] / np.sqrt(h).conj())

        np.matmul(sep_mat, mix, out=sep)
        np.power(np.abs(sep), 2, out=sep_pow)
        np.clip(sep_pow, a_min=EPS, a_max=None, out=sep_pow)

        for src in range(n_src):
            lbd = np.sqrt(np.sum(sep_pow[:, src, :]) / n_freq / n_frame)
            sep_mat[:, src, :] /= lbd
            sep_pow[:, src, :] /= lbd ** 2
            model[src] /= lbd ** 2
            basis[src] /= lbd ** 2
            # m_reci[src] *= lbd ** 2

        # convergence criterion
        Obj = np.sum(sep_pow.swapaxes(0, 1) / model + np.log(model))\
            - 2 * n_frame * np.sum(np.log(np.abs(np.linalg.det(sep_mat))))
        if np.isnan(Obj):
            flag = True
        dObj = abs(Obj - pObj) / abs(Obj)
        pObj = Obj
        if dObj < convg_tol:
            break

        # display infomation
        if t % 10 == 0:
            print('Iterations: {}, Objective: {}, dObj: {}'.format(t, Obj, dObj))

    for src in range(n_src):
        E = np.zeros((n_src, n_src))
        E[src, src] = 1
        tmp = np.linalg.inv(sep_mat) @ (E @ sep_mat) @ mix
        sep[:, src, :] = tmp[:, 1, :]

    return sep, flag


def myilrma(mix, n_iter, n_basis=2, convg_tol=1e-6, return_matrix=False):
    """Implementation of My NMF-IVA in matlab
    It only works in determined case (n_sources == n_channels).

    Args:
        mix (numpy.ndarray): (n_frequencies, n_channels, n_frames)
            STFT representation of the observed signal.
        n_iter (int): Number of iterations.
        n_basis (int): Number of basis in the NMF model.
        proj_back (bool): If use back-projection technique.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Tuple of separated signal and
            separation matrix. The shapes of separated signal and separation
            matrix are (n_frequencies, n_sources, n_frames) and
            (n_frequencies, n_sources, n_channels), respectively.
    """
    # ab_coeff = 0.005
    # alpha = 0.1
    # beta = -0.1
    flag = False
    n_freq, n_src, n_frame = mix.shape

    sep_mat = np.stack([np.eye(n_src, dtype=mix.dtype) for _ in range(n_freq)])
    sep_mat_tmp = sep_mat.swapaxes(1, 2).conj()
    basis = np.random.uniform(low=0.999, high=1.001, size=(n_src, n_freq, n_basis))
    act = np.random.uniform(low=0.999, high=1.001, size=(n_src, n_basis, n_frame))
    sep = sep_mat @ mix
    sep_abs = np.abs(sep)
    np.clip(sep_abs, a_min=EPS**0.5, a_max=None, out=sep_abs)
    sep_pow = np.power(sep_abs, 2)  # (n_freq, n_src, n_frame)
    model = basis @ act  # (n_src, n_freq, n_frame)
    m_reci = 1 / (model + EPS)

    eye = np.tile(np.eye(n_src), (n_freq, 1, 1))

    # time_per_iter = 0.0

    pObj = np.inf
    for t in range(n_iter):
        # start = time.time()

        for src in range(n_src):
            mix_acvd = mix * m_reci[src, :, None, :]
            u_mat = mix_acvd @ mix.swapaxes(1, 2).conj() / n_frame
            # use AB divergence
            # ab_acvd = np.sum(sep_abs[:, src, :]**beta, 0, keepdims=True) * (sep_abs[:, src, :]**(alpha - 2)) * alpha / 2 + \
            #     np.sum(sep_abs[:, src, :]**alpha, 0, keepdims=True) * (sep_abs[:, src, :]**(beta - 2)) * beta / 2 - \
            #     sep_abs[:, src, :]**(alpha + beta - 2) * (n_freq - 1) * (alpha + beta) / 2
            # ab_mix_acvd = mix * ab_acvd[:, None, :] / (alpha * beta * n_freq**2 * n_frame)
            # m_mat = ab_coeff * ab_mix_acvd @ mix.swapaxes(1, 2).conj()
            fin_mat = u_mat  # - m_mat
            h = sep_mat @ fin_mat + \
                0.001 * np.min(np.abs(sep_mat @ fin_mat), axis=(1, 2))[:, None, None] * eye
            h = np.linalg.inv(h)
            sep_mat_tmp[:, :, src] = h[:, :, src].copy()
            norm_fac = ((sep_mat_tmp[:, :, src, None].conj().transpose(0, 2, 1) @
                        u_mat @ sep_mat_tmp[:, :, src, None])**(-0.5)).squeeze(2)
            sep_mat_tmp[:, :, src] = sep_mat_tmp[:, :, src] * norm_fac
            # sep_mat[:, src, :] = sep_mat_tmp[:, :, src].conj()

        sep_mat = sep_mat_tmp.swapaxes(1, 2).conj()
        # sep_mat = sep_mat / np.sqrt(np.sum(np.abs(sep_mat)**2, axis=2, keepdims=True))

        np.matmul(sep_mat, mix, out=sep)
        np.power(np.abs(sep), 2, out=sep_pow)
        scale_fac = np.sqrt(np.mean(sep_pow, axis=(0, 2)))
        sep_mat = sep_mat / scale_fac[None, :, None]
        sep = sep / scale_fac[None, :, None]
        np.power(np.abs(sep), 2, out=sep_pow)
        np.clip(sep_pow, a_min=EPS, a_max=None, out=sep_pow)
        sep_abs = np.sqrt(sep_pow)
        model = model / scale_fac[:, None, None] ** 2
        basis = basis / scale_fac[:, None, None] ** 2
        m_reci = 1 / (model + EPS)

        h = basis.swapaxes(1, 2) @ (sep_pow.swapaxes(0, 1) * m_reci**2)
        h /= basis.swapaxes(1, 2) @ m_reci
        h = np.sqrt(h, out=h)
        act *= h
        model = basis @ act
        m_reci = 1 / (model + EPS)

        h = (sep_pow.swapaxes(0, 1) * m_reci**2) @ act.swapaxes(1, 2)
        h /= m_reci @ act.swapaxes(1, 2)
        h = np.sqrt(h, out=h)
        basis *= h
        model = basis @ act
        m_reci = 1 / (model + EPS)

        # end = time.time()
        # time_per_iter += end - start

        # convergence criterion
        Obj = np.sum(sep_pow.swapaxes(0, 1) * m_reci + np.log(model + EPS))\
            - 2 * n_frame * np.sum(np.log(np.abs(np.linalg.det(sep_mat))))
        if np.isnan(Obj):
            flag = True
        dObj = abs(Obj - pObj) / abs(Obj)
        pObj = Obj
        if dObj < convg_tol:
            break

        # display infomation
        if t % 10 == 0:
            print('Iterations: {}, Objective: {}, dObj: {}'.format(t, Obj, dObj))

    for src in range(n_src):
        if return_matrix:
            sep_mat_fin = np.zeros_like(sep_mat)
            for k in range(n_freq):
                sep_mat_fin[k, :, :] = \
                    np.diag(np.diag(np.linalg.inv(sep_mat[k, :, :]))) @ sep_mat[k, :, ]
                sep = sep_mat_fin @ mix
        else:
            E = np.zeros((n_src, n_src))
            E[src, src] = 1
            tmp = np.linalg.inv(sep_mat) @ (E @ sep_mat) @ mix
            sep[:, src, :] = tmp[:, 1, :]

    return (sep, flag) if not return_matrix else (sep, flag, sep_mat_fin)


if __name__ == "__main__":

    import fnmatch
    import os
    import librosa as rosa
    import mir_eval.separation as bss_eval
    # import pandas as pd
    import soundfile as sf

    # df = pd.DataFrame(columns=['Filename', 'SDR_ori', 'SDR', 'SIR_ori', 'SIR', 'perm'])

    # mix_wav_path = '/home/user/zhaoyi.gu/mnt/g2/PROJECT/MVAE_w_embds_data/mixture/vctk_90_noshuffle/p273-0_p284-0_snr-0_t60-0.15_mix.wav' # src_wav_path = '/home/user/zhaoyi.gu/mnt/g2/PROJECT/MVAE_w_embds_data/mixture/vctk_90_noshuffle/p273-0_p284-0_src.wav' # mix_wav_path = '/home/user/zhaoyi.gu/mnt/g2/PROJECT/MVAE_w_embds_data/mixture/vctk_target-11_noise-11_noshuffle/p241-0_p234-0_snr-0_t60-0.15_mix.wav' # src_wav_path = '/home/user/zhaoyi.gu/mnt/g2/PROJECT/MVAE_w_embds_data/mixture/vctk_target-11_noise-11_noshuffle/p241-0_p234-0_src.wav'
    # output_path = '/home/user/zhaoyi.gu/mnt/g2/PROJECT/MVAE_w_embds_data/output/ilrma_ab_div/'
    mix_wav_path_1 = '/data/hdd0/zhaoyigu/PROJECT/echo_split_026.wav'
    mix_wav_path_2 = '/data/hdd0/zhaoyigu/PROJECT/refer_split_026.wav'
    path_nearend = '/data/hdd0/zhaoyigu/PROJECT/refer_split_353.wav'
    mix_path = '/data/hdd0/zhaoyigu/PROJECT/stack_2.wav'
    # mix_path = '/data/hdd0/zhaoyigu/DATASET/mixspeechpian30s_anechamber_13545_0.1.wav'
    output_path = '/data/hdd0/zhaoyigu/PROJECT/out_2.wav'
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(output_path)

    # audio_1, _ = rosa.load(mix_wav_path_1, sr=16000, mono=True)
    # audio_2, _ = rosa.load(mix_wav_path_2, sr=16000, mono=True)
    # audio_3, _ = rosa.load(path_nearend, sr=16000, mono=True)
    
    # audio_1 = audio_1 / np.max(np.abs(audio_1))
    # audio_2 = audio_2 / np.max(np.abs(audio_2))
    
    # audio_3 = audio_3[:audio_1.shape[0]]
    # audio_3 = audio_3 * np.std(audio_1) * 0.3 / np.std(audio_3)
    # audio_observe = audio_1 + audio_3
    # audio_observe = audio_observe / np.max(np.abs(audio_observe))
    # audio = np.stack((audio_observe, audio_2), axis=0)
    # sf.write(output_path[:-4] + 'mix.wav', audio.T, 16000)
    audio, _ = rosa.load(mix_path, sr=16000, mono=False)
    # audio = audio[:, 5 * 16000:]
    # audio = audio[0:2,:]
    # src, _ = rosa.load(src_wav_path, sr=16000, mono=False)
    mix_spec = np.stack([rosa.stft(np.asfortranarray(x_c), n_fft=1024, hop_length=256) for x_c in audio], axis=1)
    sep_spec, _ = myilrma(mix_spec, 1000)
    sep = np.stack([rosa.istft(sep_spec[:, ch, :], hop_length=256) for ch in range(0, 2)], axis=0)
    sf.write(output_path, sep.T, 16000)

    # wav_list = list(filter(lambda x: fnmatch.fnmatch(x, '*.wav'), os.listdir(mix_wav_path)))

    # for wav_name in wav_list:

    #     print(wav_name)

    #     mix_wav_name = os.path.join(mix_wav_path, wav_name)
    #     src_wav_name = os.path.join(src_wav_path, wav_name)

    #     x, _ = rosa.load(mix_wav_name, sr=16000, mono=False)
    #     s, _ = rosa.load(src_wav_name, sr=16000, mono=False)

    #     mix_spec = np.stack([rosa.stft(np.asfortranarray(x_c), n_fft=1024, hop_length=256) for x_c in x], axis=1)

    #     sep_spec, _ = myilrma(mix_spec, 400)
    #     sep = np.stack([rosa.istft(sep_spec[:, ch, :], hop_length=256) for ch in range(0, s.shape[0])], axis=0)
    #     sf.write(output_path + wav_name, sep.T, 16000)

    #     sdr_ori, sir_ori, _, _ = bss_eval.bss_eval_sources(s, x)
    #     sdr, sir, _, perm = bss_eval.bss_eval_sources(s, sep)

    #     np.set_printoptions(precision=2)
    #     print('SIR_ori: {}, SIR: {}'.format(sir_ori, sir))
    #     print('SDR_ori: {}, SDR：{}'.format(sdr_ori, sdr))

    #     df = df.append(pd.DataFrame({
    #                                 'Filename': [wav_name[0:-4] for _ in range(2)],
    #                                 'SDR_ori': sdr_ori,
    #                                 'SDR': sdr,
    #                                 'SIR_ori': sir_ori,
    #                                 'SIR': sir,
    #                                 'perm': perm
    #                                 }))
    #     df.to_csv(output_path + 'myilrma.csv', index=None)










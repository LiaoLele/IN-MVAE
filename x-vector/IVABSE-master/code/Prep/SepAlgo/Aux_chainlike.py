import numpy as np


def chainlike(mix, n_iter=1000, clique_bins=256, clique_hop=128, convg_tol=1e-6):

    flag = False
    n_freq, n_src, n_frame = mix.shape
    eye = np.tile(np.eye(n_src), (n_freq, 1, 1))

    num_clique_bins = (n_freq - clique_bins) // clique_hop + 1
    print(num_clique_bins)

    sep_mat = np.stack([np.eye(n_src, dtype=mix.dtype) for _ in range(n_freq)])
    sep_mat_tmp = sep_mat.swapaxes(1, 2).conj()
    sep = sep_mat @ mix
    sep_pow = np.power(np.abs(sep), 2)  # (n_freq, n_src, n_frame)
    stat_1_c = np.zeros((num_clique_bins, n_src, n_frame))
    stat_1_f = np.zeros((n_freq, n_src, n_frame))
    for i in range(num_clique_bins):
        if i == num_clique_bins - 1:
            stat_1_c[i, :, :] = 1 / (np.sqrt(np.sum(sep_pow[i * clique_hop:, :, :], axis=0)) + 1e-7)
            stat_1_f[i * clique_hop:, :, :] += stat_1_c[i, None, :, :]
        else:
            stat_1_c[i, :, :] = 1 / (np.sqrt(np.sum(sep_pow[i * clique_hop: i * clique_hop + clique_bins, :, :], axis=0)) + 1e-7)
            stat_1_f[i * clique_hop: i * clique_hop + clique_bins, :, :] += stat_1_c[i, None, :, :]

    pObj = np.inf
    for t in range(n_iter):
        for src in range(n_src):
            # h = stat_1_f[:, src, :, None] @ np.ones((1, n_src))
            # h = mix.conj() @ (mix.swapaxes(1, 2) * h)
            # u_mat = h.swapaxes(1, 2) / n_frame
            # h = sep_mat @ u_mat + 1e-9 * eye
            # sep_mat[:, src, :] = np.linalg.solve(h, eye[:, :, src]).conj()
            # h = sep_mat[:, src, None, :] @ u_mat
            # h = (h @ sep_mat[:, src, :, None].conj()).squeeze(2)
            # sep_mat[:, src, :] = (sep_mat[:, src, :] / np.sqrt(h).conj())

            mix_acvd = mix * stat_1_f[:, src, None, :]
            u_mat = mix_acvd @ mix.swapaxes(1, 2).conj() / n_frame
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
        sep_mat = sep_mat / np.sqrt(np.sum(np.abs(sep_mat)**2, axis=2, keepdims=True))

        np.matmul(sep_mat, mix, out=sep)
        np.power(np.abs(sep), 2, out=sep_pow)
        stat_1_f = np.zeros((n_freq, n_src, n_frame))
        for i in range(num_clique_bins):
            if i == num_clique_bins - 1:
                stat_1_c[i, :, :] = 1 / (np.sqrt(np.sum(sep_pow[i * clique_hop:, :, :], axis=0)) + 1e-7)
                stat_1_f[i * clique_hop:, :, :] += stat_1_c[i, None, :, :]
            else:
                stat_1_c[i, :, :] = 1 / (np.sqrt(np.sum(sep_pow[i * clique_hop: i * clique_hop + clique_bins, :, :], axis=0)) + 1e-7)
                stat_1_f[i * clique_hop: i * clique_hop + clique_bins, :, :] += stat_1_c[i, None, :, :]

        Obj = np.sum(stat_1_c) - 2 * n_frame * np.sum(np.log(np.abs(np.linalg.det(sep_mat))))
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


def chainlike_prob(mix, n_iter=1000, clique_bins=256, clique_hop=128, convg_tol=1e-6):

    flag = False
    n_freq, n_src, n_frame = mix.shape
    eye = np.tile(np.eye(n_src), (n_freq, 1, 1))

    num_clique_bins = (n_freq - clique_bins) // clique_hop + 1

    sep_mat = np.stack([np.eye(n_src, dtype=mix.dtype) for _ in range(n_freq)])
    sep_mat_tmp = sep_mat.swapaxes(1, 2).conj()
    sep = sep_mat @ mix
    sep_pow = np.power(np.abs(sep), 2)  # (n_freq, n_src, n_frame)
    stat_1_c = np.zeros((num_clique_bins, n_src, n_frame))
    for i in range(num_clique_bins):
        if i == num_clique_bins - 1:
            stat_1_c[i, :, :] = 1 / np.sqrt(np.sum(sep_pow[i * clique_hop:, :, :], axis=0))
        else:
            stat_1_c[i, :, :] = 1 / np.sqrt(np.sum(sep_pow[i * clique_hop: i * clique_hop + clique_bins, :, :], axis=0))
    scale = np.sum(stat_1_c, axis=0, keepdims=True)

    pObj = np.inf
    for t in range(n_iter):
        for src in range(n_src):
            scale_p = np.tile(scale, (n_freq, 1, 1))
            h = scale_p[:, src, :, None] @ np.ones((1, n_src))
            h = mix.conj() @ (mix.swapaxes(1, 2) * h)
            u_mat = h.swapaxes(1, 2) / n_frame
            h = sep_mat @ u_mat + 1e-9 * eye
            sep_mat[:, src, :] = np.linalg.solve(h, eye[:, :, src]).conj()
            h = sep_mat[:, src, None, :] @ u_mat
            h = (h @ sep_mat[:, src, :, None].conj()).squeeze(2)
            sep_mat[:, src, :] = (sep_mat[:, src, :] / np.sqrt(h).conj())

            # mix_acvd = mix * scale[:, src, None, :]
            # u_mat = mix_acvd @ mix.swapaxes(1, 2).conj() / n_frame
            # fin_mat = u_mat  # - m_mat
            # h = sep_mat @ fin_mat + \
            #     0.001 * np.min(np.abs(sep_mat @ fin_mat), axis=(1, 2))[:, None, None] * eye
            # h = np.linalg.inv(h)
            # sep_mat_tmp[:, :, src] = h[:, :, src].copy()
            # norm_fac = ((sep_mat_tmp[:, :, src, None].conj().transpose(0, 2, 1) @
            #             u_mat @ sep_mat_tmp[:, :, src, None])**(-0.5)).squeeze(2)
            # sep_mat_tmp[:, :, src] = sep_mat_tmp[:, :, src] * norm_fac
            # sep_mat[:, src, :] = sep_mat_tmp[:, :, src].conj()

        np.matmul(sep_mat, mix, out=sep)
        np.power(np.abs(sep), 2, out=sep_pow)
        for i in range(num_clique_bins):
            if i == num_clique_bins - 1:
                stat_1_c[i, :, :] = 1 / np.sqrt(np.sum(sep_pow[i * clique_hop:, :, :], axis=0))
            else:
                stat_1_c[i, :, :] = 1 / np.sqrt(np.sum(sep_pow[i * clique_hop: i * clique_hop + clique_bins, :, :], axis=0))
        scale = np.sum(stat_1_c, axis=0, keepdims=True)

        Obj = np.sum(stat_1_c) - 2 * n_frame * np.sum(np.log(np.abs(np.linalg.det(sep_mat))))
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
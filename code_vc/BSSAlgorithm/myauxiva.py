import numpy as np
from .pca import pca
from .projection_back import project_back

EPS = 3e-16

def auxiva(mix, n_iter=100, convg_tol=1e-6):
    """
    Args:
        mix (numpy.ndarray): (n_frequencies, n_channels, n_frames)
            STFT representation of the observed signal.
        n_iter (int): Number of iterations.
        
        proj_back (bool): If use back-projection technique.

    Returns:
        numpy.ndarray: separated signal. The shapes of separated signal is 
        (n_frequencies, n_sources, n_frames).
    """
    n_freq, n_src, n_frame = mix.shape
    mix_ori = mix
    mix, W_pca = pca(mix.transpose(2, 0, 1), n_src=n_src, return_filters=True)
    # W_pca: (n_freq, n_src, n_src)
    mix = mix.transpose(1, 2, 0)
    sep_mat = np.stack([np.eye(n_src, dtype=mix.dtype) for _ in range(n_freq)])
    

    eye = np.tile(np.eye(n_src), (n_freq, 1, 1))

    pObj = np.inf
    for t in range(n_iter):
        for src in range(n_src):
            sep = sep_mat @ mix
            sep_pow = np.power(np.abs(sep), 2)  # (n_freq, n_src, n_frame)
            np.clip(sep_pow, a_min=EPS, a_max=None, out=sep_pow)
            model = np.power(np.sum(sep_pow, axis=0, keepdims=True), 0.5)
            m_reci = 1 / model.transpose(1, 0, 2) # (n_src, n_freq, n_frame)

            h = m_reci[src, :, :, None] @ np.ones((1, n_src)) # (n_freq, n_frame, n_src)
            h = mix.conj() @ (mix.swapaxes(1, 2) * h)
            # (n_freq, n_src, n_src)
            u_mat = h.swapaxes(1, 2) / n_frame
            h = sep_mat @ u_mat + EPS * eye
            sep_mat[:, src, :] = np.linalg.solve(h, eye[:, :, src]).conj()
            h = sep_mat[:, src, None, :] @ u_mat
            h = (h @ sep_mat[:, src, :, None].conj()).squeeze(2)
            sep_mat[:, src, :] = (sep_mat[:, src, :] / np.sqrt(h).conj())

        # convergence criterion
        Obj = -np.sum(np.log(np.abs(np.linalg.det(sep_mat))))/(n_frame*n_freq)
        dObj = abs(Obj - pObj) / abs(Obj)
        pObj = Obj
        if dObj < convg_tol:
            break

        # display infomation
        if t % 10 == 0:
            print('Iterations: {}, Objective: {}, dObj: {}'.format(t, Obj, dObj))

    # sep = project_back(sep.transpose(2, 0, 1), mix[:, 0, :].transpose(1, 0)).transpose(1, 2, 0)

    sep_mat = sep_mat @ W_pca
    for src in range(n_src):
        E = np.zeros((n_src, n_src))
        E[src, src] = 1
        tmp = np.linalg.inv(sep_mat) @ (E @ sep_mat) @ mix_ori
        sep[:, src, :] = tmp[:, 1, :]

    return sep

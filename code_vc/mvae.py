import numpy as np
import torch
from BSSAlgorithm.ilrma import myilrma
from BSSAlgorithm.common import projection_back
from BSSAlgorithm.ilrma import ilrma

EPS = 1e-9


def mvae(mix, model, n_iter, device, proj_back=True, return_sigma=False):
    """Implementation of Multichannel Conditional VAE.
    It only works in determined case (n_sources == n_channels).

    Args:
        mix (numpy.ndarray): (n_frequencies, n_channels, n_frames)
            STFT representation of the observed signal.
        model (cvae.CVAE): Trained Conditional VAE model.
        n_iter (int): Number of iterations.
        device (torch.device): Device used for computation.
        proj_back (bool): If use back-projection technique.
        return_sigma (bool): If also return estimated power spectrogram for
            each speaker.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Tuple of separated signal and
            separation matrix. The shapes of separated signal and separation
            matrix are (n_frequencies, n_sources, n_frames) and
            (n_frequencies, n_sources, n_channels), respectively.
    """
    n_freq, n_src, n_frame = mix.shape

    # sep, sep_mat = ilrma(mix, n_iter=30, n_basis=2)
    sep, _, sep_mat = myilrma(mix, 30, return_matrix=True)
    sep_pow = np.power(np.abs(sep), 2)  # (n_freq, n_src, n_frame)
    c = torch.full((n_src, model.n_speakers), 1 / model.n_speakers,
                   device=device, requires_grad=True)
    log_g = torch.full((n_src, 1, 1), model.log_g.item(), device=device)

    with torch.no_grad():
        sep_pow_tensor = torch.from_numpy(sep_pow).transpose(0, 1).to(device) # (n_src, n_freq, n_frame)
        sep_pow_tensor.clamp_(EPS)
        z, _ = model.encode(sep_pow_tensor, c)
        z = z.to(device)
        sigma_sq = (model.decode(z, c).to(device) + log_g).exp()
        sigma_sq.clamp_(min=EPS)
        sigma_reci = (1 / sigma_sq).cpu().numpy() 
        
    z.requires_grad = True

    eye = np.tile(np.eye(n_src), (n_freq, 1, 1))
    pObj = np.Inf
    for ii in range(n_iter):
        for src in range(n_src):
            h = sigma_reci[src, :, :, None] @ np.ones((1, n_src)) #(n_freq,n_frame,n_src)
            h = mix.conj() @ (mix.swapaxes(1, 2) * h)  #(n_freq,n_src,n_src)
            u_mat = h.swapaxes(1, 2) / n_frame
            h = sep_mat @ u_mat + EPS * eye
            sep_mat[:, src, :] = np.linalg.solve(h, eye[:, :, src]).conj()
            h = sep_mat[:, src, None, :] @ u_mat
            h = (h @ sep_mat[:, src, :, None].conj()).squeeze(2) 
            sep_mat[:, src, :] = (sep_mat[:, src, :] / np.sqrt(h).conj())

        np.matmul(sep_mat, mix, out=sep)
        np.power(np.abs(sep), 2, out=sep_pow)
        np.clip(sep_pow, a_min=EPS, a_max=None, out=sep_pow) # shape = (nfreq, nsrc, nframe)

        optimizer = torch.optim.Adam((z, c), lr=1e-3)
        sep_pow_tensor = torch.from_numpy(sep_pow).to(device).transpose(0, 1) # shape = (nsrc, nfreq, nframe)
        for _ in range(50): # 50 epoch
            log_sigma_sq = model.decode(z, torch.softmax(c, dim=1)).to(device) + log_g
            loss = torch.sum(
                log_sigma_sq + (sep_pow_tensor.log() - log_sigma_sq).exp()) # IVA's cost func. except for logdet(W) term
            model.zero_grad()
            optimizer.zero_grad()        
            loss.backward()
            optimizer.step()


        with torch.no_grad():
            sigma_sq = (model.decode(z, torch.softmax(c, dim=1)).to(device) + log_g).exp()
            lbd = torch.sum(sep_pow_tensor / sigma_sq, dim=(1, 2))
            lbd = lbd / n_freq / n_frame / log_g.squeeze(2).squeeze(1).exp()
            log_g[:, 0, 0] += torch.log(lbd)
            sigma_sq *= lbd.unsqueeze(1).unsqueeze(2)
            sep_mat *= lbd.unsqueeze(0).unsqueeze(2).cpu().numpy()

            sigma_reci = (1 / sigma_sq).cpu().numpy()
            sigma_sq = sigma_sq.cpu().numpy()
        Obj = np.sum(- sep_pow.swapaxes(0, 1) * sigma_reci - np.log(sigma_sq))\
            + 2 * n_frame * np.sum(np.log(np.abs(np.linalg.det(sep_mat))))
        dObj = np.abs(Obj - pObj) / np.abs(Obj)
        pObj = Obj
        if dObj < 1e-5:
            break
        # display infomation
        if ii % 10 == 0:
            print('MVAE-Iterations: {}, Objective: {}, dObj: {}'.format(ii, Obj, dObj))

    # Back-projection technique
    if proj_back:
        z = projection_back(sep, mix[:, 0, :])
        sep *= np.conj(z[:, :, None])

    if return_sigma:
        return sep, sep_mat, sigma_sq.cpu().numpy()
    else:
        return sep, sep_mat

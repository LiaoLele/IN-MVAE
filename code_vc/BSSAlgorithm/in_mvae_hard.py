import torch
import numpy as np
from BSSAlgorithm.ilrma import myilrma
from PreProcess.data_utils import _mel_to_linear_matrix
from config import cfgs
from PreProcess.data_utils import Spectrogram
import librosa
import torch.nn.functional as F

EPS = 1e-9


def MVAEIVA(mix, target_embd, model, n_iter=600, device=None, convg_tol=1e-5,
            ilrma_init=False, alternate_align_init=False, latent_meth='bp', nsamp=None, able2convg=False):
    """ Args """
    """ 
    `mix`: [n_freq, n_src, n_frame]
    `target_embd`: [d_embd, ]
    `latent_meth`: 'bp'/'encoder'
    """
    
    spkr_embd_input = target_embd
    # target_embd = torch.ones((0, cfgs.NetInput.n_embedding)).to(cfgs.general.device)
    
    model = model.eval()
    assert len(mix.shape) == 3
    if target_embd.dim() == 1:
        target_embd = target_embd.unsqueeze(0)
    if latent_meth == 'encoder' and nsamp is None:
        print("Only use the mean value of decoder output! Activate sampling by setting `nsamp`!")
    if latent_meth == 'encoder_update' and nsamp is None:
        print("Set nsamp to default value 10!")
        nsamp = 10
        
    nan_flag = False
    n_freq, n_src, n_frame = mix.shape
    d_embd = target_embd.shape[1]
    device = target_embd.device
    eye = np.tile(np.eye(n_src), (n_freq, 1, 1))
    mel_basis = torch.from_numpy(librosa.filters.mel(cfgs.sigproc.sr, int(cfgs.sigproc.sr*cfgs.sigproc.stft_len), n_mels=cfgs.sigproc.nmels)).to(device).unsqueeze(0)
    m = _mel_to_linear_matrix(cfgs.sigproc.sr, int(cfgs.sigproc.sr*cfgs.sigproc.stft_len), cfgs.sigproc.nmels)

    # Initialization
    if ilrma_init:
        sep_spec, _, sep_mat = myilrma(mix, 30, return_matrix=True)
    else:
        sep_mat = np.tile(np.eye(n_src), (n_freq, 1, 1)).astype(np.complex64)
        if alternate_align_init:
            for i in range(1, n_freq, 2):
                sep_mat[i, :, :] = np.array([[0, 1], [1, 0]], dtype=np.complex64)
        sep_spec = sep_mat @ mix
    sep_spec_pow = np.power(np.abs(sep_spec), 2)
    sep_spec_pow_tensor = torch.from_numpy(sep_spec_pow).to(device).transpose(0, 1).float()
    sep_spec_float = sep_spec_pow_tensor.pow(0.5)
    sep_spec_mel = torch.matmul(mel_basis, sep_spec_float)
    sep_spec_mel = 20 * torch.log10(sep_spec_mel)
    sep_spec_mel = (sep_spec_mel - 20 + 100) / 100
    if target_embd.shape[0] < n_src:
        interference_embd = 1 / d_embd * torch.ones(n_src - target_embd.shape[0], *target_embd.shape[1:]).to(device)
    elif target_embd.shape[0] == n_src:
        interference_embd = torch.ones(0, *target_embd.shape[1:]).to(device)
    elif target_embd.shape[0] == 0:
        interference_embd = 1 / d_embd * torch.ones(n_src, *target_embd.shape[1:]).to(device)
    log_factor = torch.full((n_src, 1, 1), 1.0, device=device)
    with torch.no_grad():
        spkr_embd = torch.cat((target_embd, interference_embd), dim=0)
        content_embd, _ = model.content_encoder(sep_spec_mel)
        mel = model.decoder(content_embd, spkr_embd)
        mel = (mel * 100) - 100 + 20
        # m: [nfreq, nmel]
        mel = torch.pow(10.0, mel * 0.05) # [nsou, nmel, nframe]
        mag = torch.matmul(m, mel)
        sep_var = ((mag**2).log() + log_factor).exp()[:, :n_freq, :n_frame]
        sep_reci = (1 / sep_var).detach().cpu().numpy()
    content_embd.requires_grad = True
    interference_embd.requires_grad = True
    target_embd.requires_grad = False

    # iteration
    pObj = np.Inf
    for t in range(n_iter):
        np.matmul(sep_mat, mix, out=sep_spec)
        np.power(np.abs(sep_spec), 2, out=sep_spec_pow)
        np.clip(sep_spec_pow, a_min=EPS, a_max=None, out=sep_spec_pow)
        sep_spec_pow_tensor = torch.from_numpy(sep_spec_pow).to(device).transpose(0, 1).float()
        #(n_src,n_freq,n_frame)
        sep_spec_float = sep_spec_pow_tensor.pow(0.5)
        sep_spec_mel = torch.matmul(mel_basis, sep_spec_float)
        sep_spec_mel = 20 * torch.log10(sep_spec_mel)
        sep_spec_mel = (sep_spec_mel - 20 + 100) / 100
        if latent_meth.startswith('bp'):
            if latent_meth.startswith('bp_encoderinit'):
                with torch.no_grad():
                    content_embd, _ = model.content_encoder(sep_spec_mel)
                content_embd.requires_grad = True

                # content_embd.is_leaf = True
                # content_embd, _ = model.content_encoder(sep_spec_mel)
                # content_embd.requires_grad = True
                # sep_spec_mel.requires_grad = False
            optimizer = torch.optim.Adam((content_embd, interference_embd), lr=1e-3)
            # optimizer = torch.optim.SGD((content_embd,), lr=1e-3)
            for _ in range(10):
                assert interference_embd.requires_grad is True
                assert content_embd.requires_grad is True
                assert target_embd.requires_grad is False
                assert interference_embd.is_leaf is True
                assert content_embd.is_leaf is True
                assert target_embd.is_leaf is True
                spkr_embd = torch.cat((target_embd, interference_embd), dim=0)
                mel_ = model.decoder(content_embd, spkr_embd)[:, :n_freq, :n_frame]
                mel = (mel_ * 100) - 100 + 20
                # m: [nfreq, nmel]
                mel = torch.pow(10.0, mel * 0.05) # [nsou, nmel, nframe]
                mag = torch.matmul(m, mel)
                log_sep_var = (mag**2).log() + log_factor
                if latent_meth.endswith('mle'):
                    loss = torch.sum(log_sep_var + (sep_spec_pow_tensor.log() - log_sep_var).exp())
                elif latent_meth.endswith('map'):
                    loss = torch.sum(log_sep_var + (sep_spec_pow_tensor.log() - log_sep_var).exp()) - torch.sum(content_embd**2)
                elif latent_meth.endswith('l1'):
                    loss = F.l1_loss(mel_, sep_spec_mel, reduction='sum')
                elif latent_meth.endswith('mse'):
                    loss = F.mse_loss(mel_, sep_spec_mel, reduction='sum')
                model.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        elif latent_meth.startswith('encoder'):
            if latent_meth == 'encoder_update':
                optimizer = torch.optim.SGD(list(model.content_encoder.parameters()) + list(interference_embd), lr=1e-3)
                for _ in range(10):
                    content_mu, content_log_var = model.content_encoder(sep_spec_mel)
                    content_embd = content_log_var.new_empty(nsamp, *content_log_var.size()).normal_(0, 1).transpose(0, 1)  # [n_src, nsamp, d_content_mu]
                    content_embd = content_embd * torch.exp(content_log_var.unsqueeze(1) * 0.5) + content_mu.unsqueeze(1)
                    content_embd = content_embd.reshape(-1, *content_embd.shape[2:])  # [nsrc * nsamp, d_content_mu]

                    spkr_embd = torch.cat((target_embd, interference_embd), dim=0)
                    spkr_embd_tmp = spkr_embd.repeat_interleave(int(content_embd.shape[0] / n_src), dim=0)
                    mel = model.decoder(content_embd, spkr_embd_tmp)[:, :n_freq, :n_frame]
                    mel = (mel * 100) - 100 + 20
                    # m: [nfreq, nmel]
                    mel = torch.pow(10.0, mel * 0.05) # [nsou, nmel, nframe]
                    mag = torch.matmul(m, mel)
                    sep_var = (mag**2).reshape(n_src, -1, n_freq, n_frame).clamp_min(EPS)
                    L_rec = torch.sum((sep_spec_pow_tensor.log() + torch.mean(1 / sep_var, dim=1).log() - log_factor).exp() \
                                      + log_factor + torch.mean(sep_var.log(), dim=1))
                    L_kl = 0.5 * torch.sum(content_mu.pow(2) + content_log_var.exp() - content_log_var)
                    loss = L_rec + L_kl
                    model.zero_grad()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                content_mu, content_log_var = model.content_encoder(sep_spec_mel)
                if nsamp is not None:
                    content_embd = content_log_var.new_empty((nsamp, *content_log_var.size())).normal_(0, 1).transpose(0, 1)  # [n_src, nsamp, ...]
                    content_embd = content_embd * torch.exp(content_log_var.unsqueeze(1) * 0.5) + content_mu.unsqueeze(1)
                    content_embd = content_embd.reshape(-1, *content_embd.shape[2:])
                elif nsamp is None:
                    content_embd = content_mu

        # update g
        with torch.no_grad():
            spkr_embd = torch.cat((target_embd, interference_embd), dim=0)
            spkr_embd_tmp = spkr_embd.repeat_interleave(int(content_embd.shape[0] / n_src), dim=0)
            mel = model.decoder(content_embd, spkr_embd_tmp)[:, :n_freq, :n_frame]
            mel = (mel * 100) - 100 + 20
            # m: [nfreq, nmel]
            mel = torch.pow(10.0, mel * 0.05) # [nsou, nmel, nframe]
            mag = torch.matmul(m, mel)
            sep_logvar_ori = (mag**2).log()
            sep_var_ori = sep_logvar_ori.exp().reshape(n_src, -1, *sep_logvar_ori.shape[1:])
            sep_reci_ori = torch.mean(1 / torch.clamp_min(sep_var_ori, EPS), dim=1)
            log_factor[:, 0, 0] = torch.mean(sep_spec_pow_tensor * sep_reci_ori, dim=(1, 2)).log()

            log_factor_tmp = log_factor.repeat_interleave(int(content_embd.shape[0] / n_src), dim=0)
            sep_var = (sep_logvar_ori + log_factor_tmp).exp().reshape(n_src, -1, n_freq, n_frame).cpu().numpy()
            sep_var = np.clip(sep_var, a_min=EPS, a_max=None)   # [nsrc, nsamp, ...]
            sep_reci = np.mean(1 / sep_var, axis=1)

        # Update W
        for n in range(n_src):
            h = sep_reci[n, :, :, None] @ np.ones((1, n_src))
            h = mix.conj() @ (mix.swapaxes(1, 2) * h)
            u_mat = h.swapaxes(1, 2) / n_frame
            h = sep_mat @ u_mat + 0.001 * np.min(np.abs(sep_mat @ u_mat), axis=(1, 2))[:, None, None] * eye
            sep_mat[:, n, :] = np.linalg.solve(h, eye[:, :, n]).conj()
            h = sep_mat[:, n, None, :] @ u_mat
            h = (h @ sep_mat[:, n, :, None].conj()).squeeze(2)
            sep_mat[:, n, :] = (sep_mat[:, n, :] / np.sqrt(h).conj())
        sep_mat = sep_mat / np.sqrt(np.sum(np.abs(sep_mat)**2, axis=2, keepdims=True))

        # convergence criterion
        Obj = np.sum(- sep_spec_pow.swapaxes(0, 1) * sep_reci - np.mean(np.log(sep_var), axis=1))\
            + 2 * n_frame * np.sum(np.log(np.abs(np.linalg.det(sep_mat))))
        if np.isnan(Obj):
            nan_flag = True
        dObj = np.abs(Obj - pObj) / np.abs(Obj)
        pObj = Obj
        if dObj < convg_tol:
            break
        # display infomation
        # if t % 10 == 0:
        #     print('MVAE-Iterations: {}, Objective: {}, dObj: {}'.format(t, Obj, dObj))

    # Back-projection technique
    sep_mat_inv = np.linalg.inv(sep_mat)
    for n in range(n_src):
        E = np.zeros((n_src, n_src))
        E[n, n] = 1
        tmp = sep_mat_inv @ (E @ sep_mat) @ mix
        sep_spec[:, n, :] = tmp[:, 1, :]
    sep_spec_pow = np.power(np.abs(sep_spec), 2)
    sep_spec_pow_tensor = torch.from_numpy(sep_spec_pow).to(device).transpose(0, 1).float()
    sep_spec_float = sep_spec_pow_tensor.pow(0.5)
    sep_spec_mel = torch.matmul(mel_basis, sep_spec_float)
    sep_spec_mel = 20 * torch.log10(sep_spec_mel)
    sep_spec_mel = (sep_spec_mel - 20 + 100) / 100
    # spkr_embd_input = target_embd
    spkr_embd_output = model.get_speaker_embeddings(sep_spec_mel)
    # ori_mse = torch.sum(torch.sum((spkr_embd_input - spkr_embd_output).pow(2), 1).pow(0.5))
    # swap_mse = torch.sum(torch.sum((spkr_embd_input - spkr_embd_output[[1,0],:]).pow(2), 1).pow(0.5))
    _, spkr_mean_input, _ = model.decoder.conv_blocks_forward(content_embd, spkr_embd_input)
    _, spkr_mean_output, _ = model.decoder.conv_blocks_forward(content_embd, spkr_embd_output)
    ori = torch.sum((spkr_mean_input - spkr_mean_output).pow(2), 1).pow(0.5)
    swap = torch.sum((spkr_mean_input - spkr_mean_output[[1,0],:]).pow(2), 1).pow(0.5)
    sum_ori = torch.sum(ori)
    sum_swap = torch.sum(swap)
    if sum_ori > sum_swap:
        sep_spec = sep_spec[:,[1,0],:]
    perm_flag = 1
    return sep_spec, perm_flag


def MVAEOnehotIVA(mix, vae_model=None, device=None, n_iter=1000, convg_tol=1e-6,
                  ilrma_init=False, alternate_align_init=False, latent_meth='bp', nsamp=None):
    flag = False
    n_freq, n_src, n_frame = mix.shape
    eye = np.tile(np.eye(n_src), (n_freq, 1, 1))

    # Initialization
    if ilrma_init:
        sep, _, sep_mat = myilrma(mix, 30, return_matrix=True)
    else:
        sep = mix.copy()
        sep_mat = np.tile(np.eye(n_src), (n_freq, 1, 1)).astype(np.complex64)
    c = 1 / vae_model.n_embedding * torch.ones(n_src, vae_model.n_embedding).to(device).float()
    sep_pow = np.power(np.abs(sep), 2)
    log_g = torch.full((n_src, 1, 1), vae_model.log_g.item(), device=device)

    with torch.no_grad():
        sep_pow_tensor = torch.from_numpy(sep_pow).transpose(0, 1).to(device).float()
        #（n_src, n_freq, n_frame）
        sep_pow_tensor.clamp_(EPS)
        z, _ = vae_model.encode(sep_pow_tensor, c)
        # z = torch.randn_like(tmp)
        sigma_sq = (vae_model.decode(z, c) + log_g).exp()
        sigma_sq.clamp_(min=EPS)
        sigma_reci = (1 / sigma_sq).cpu().numpy()
    z.requires_grad = True
    c.requires_grad = True

    pObj = np.Inf
    for t in range(n_iter):
        for n in range(n_src):
            h = sigma_reci[n, :, :, None] @ np.ones((1, n_src))
            h = mix.conj() @ (mix.swapaxes(1, 2) * h)
            u_mat = h.swapaxes(1, 2) / n_frame
            h = sep_mat @ u_mat + 0.001 * np.min(np.abs(sep_mat @ u_mat), axis=(1, 2))[:, None, None] * eye
            sep_mat[:, n, :] = np.linalg.solve(h, eye[:, :, n]).conj()
            h = sep_mat[:, n, None, :] @ u_mat
            h = (h @ sep_mat[:, n, :, None].conj()).squeeze(2)
            sep_mat[:, n, :] = (sep_mat[:, n, :] / np.sqrt(h).conj())

        sep_mat = sep_mat / np.sqrt(np.sum(np.abs(sep_mat)**2, axis=2, keepdims=True))

        np.matmul(sep_mat, mix, out=sep)
        np.power(np.abs(sep), 2, out=sep_pow)
        np.clip(sep_pow, a_min=EPS, a_max=None, out=sep_pow)
        sep_pow_tensor = torch.from_numpy(sep_pow).to(device).transpose(0, 1)

        optimizer = torch.optim.Adam((z, c), lr=1e-3)
        for _ in range(10):
            log_sigma_sq = vae_model.decode(z, torch.softmax(c, dim=1)) + log_g
            loss = torch.sum(log_sigma_sq + (sep_pow_tensor.log() - log_sigma_sq).exp())
            vae_model.zero_grad()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        with torch.no_grad():
            sigma_sq = (vae_model.decode(z, torch.softmax(c, dim=1)) + log_g).exp()
            lbd = torch.sum(sep_pow_tensor / sigma_sq, dim=(1, 2))
            lbd = lbd / n_freq / n_frame
            log_g[:, 0, 0] += torch.log(lbd)
            sigma_sq = (vae_model.decode(z, torch.softmax(c, dim=1)) + log_g).exp()
            sigma_reci = (1 / sigma_sq).cpu().numpy()

        # convergence criterion
        Obj = np.sum(- sep_pow.swapaxes(0, 1) * sigma_reci + np.log(sigma_reci))\
            + 2 * n_frame * np.sum(np.log(np.abs(np.linalg.det(sep_mat))))
        if np.isnan(Obj):
            flag = True
        dObj = np.abs(Obj - pObj) / np.abs(Obj)
        pObj = Obj
        if dObj < convg_tol:
            break
        # display infomation
        if t % 10 == 0:
            print('MVAE-Iterations: {}, Objective: {}, dObj: {}'.format(t, Obj, dObj))

    # Back-projection technique
    sep_mat_inv = np.linalg.inv(sep_mat)
    for n in range(n_src):
        E = np.zeros((n_src, n_src))
        E[n, n] = 1
        tmp = sep_mat_inv @ (E @ sep_mat) @ mix
        sep[:, n, :] = tmp[:, 1, :]
    
    label_est = torch.softmax(c, dim=1)
    # c(n_src,n_embding)
    return sep, label_est, flag
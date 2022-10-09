import numpy as np
import torch
from SepAlgo.ilrma import myilrma


EPS = 1e-9


def spectrogram_normalize(sample):  # sample [channel, nfreq, time]
    sample_max, _ = sample.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)
    sample = sample / sample_max
    return sample


def mvae_onehot(mix, vae_model, n_iter=600, device=None, convg_tol=1e-6, ilrma_init=False):
    flag = False
    mix = mix.swapaxes(1, 2)
    n_freq, n_src, n_frame = mix.shape
    eye = np.tile(np.eye(n_src), (n_freq, 1, 1))

    # Initialization
    if ilrma_init:
        sep, _, sep_mat = myilrma(mix, 30, return_matrix=True)
    else:
        sep = mix.copy()
        sep_mat = np.tile(np.eye(n_src), (n_freq, 1, 1)).astype(np.complex64)
    c = 1 / vae_model.d_embedding * torch.ones(n_src, vae_model.d_embedding).to(device)
    sep_pow = np.power(np.abs(sep), 2)
    log_g = torch.full((n_src, 1, 1), vae_model.log_g.item(), device=device)

    with torch.no_grad():
        sep_pow_tensor = torch.from_numpy(sep_pow).transpose(0, 1).to(device)
        sep_pow_tensor.clamp_(EPS)
        z, _ = vae_model.encode(sep_pow_tensor, c)
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
            loss.backward(retain_graph=True)  #retain_graph=True
            optimizer.step()

        with torch.no_grad():
            sigma_sq = (vae_model.decode(z, torch.softmax(c, dim=1)) + log_g).exp()
            lbd = torch.sum(sep_pow_tensor / sigma_sq, dim=(1, 2))
            lbd = lbd / n_freq / n_frame
            log_g[:, 0, 0] += torch.log(lbd)
            sigma_sq = (vae_model.decode(z, torch.softmax(c, dim=1)) + log_g).exp()
            sigma_reci = (1 / sigma_sq).cpu().numpy()

            # sigma_sq = (vae_model.decode(z, torch.softmax(c, dim=1)) + log_g).exp()
            # lbd = torch.sum(sep_pow_tensor / sigma_sq, dim=(1, 2))
            # lbd = lbd / n_freq / n_frame / log_g.squeeze(2).squeeze(1).exp()
            # log_g[:, 0, 0] += torch.log(lbd)
            # sigma_sq *= lbd.unsqueeze(1).unsqueeze(2)
            # sep_mat *= lbd.unsqueeze(0).unsqueeze(2).cpu().numpy()
            # sigma_reci = (1 / sigma_sq).cpu().numpy()

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

    return sep, flag


def mvae_onehot_official(mix, vae_model, n_iter=600, device=None, convg_tol=1e-6, ilrma_init=False):
    flag = False
    mix = mix.swapaxes(1, 2)
    n_freq, n_src, n_frame = mix.shape
    eye = np.tile(np.eye(n_src), (n_freq, 1, 1))

    # Initialization
    if ilrma_init:
        sep, _, sep_mat = myilrma(mix, 30, return_matrix=True)
    else:
        sep = mix.copy()
        sep_mat = np.tile(np.eye(n_src), (n_freq, 1, 1)).astype(np.complex64)
    c = 1 / vae_model.d_embedding * torch.ones(n_src, vae_model.d_embedding).to(device)
    sep_pow = np.power(np.abs(sep), 2)
    log_g = torch.full((n_src, 1, 1), vae_model.log_g.item(), device=device)

    with torch.no_grad():
        sep_pow_tensor = torch.from_numpy(sep_pow).transpose(0, 1).to(device)
        sep_pow_tensor.clamp_(EPS)
        z, _ = vae_model.encode(sep_pow_tensor, c)
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

        np.matmul(sep_mat, mix, out=sep)
        np.power(np.abs(sep), 2, out=sep_pow)
        np.clip(sep_pow, a_min=EPS, a_max=None, out=sep_pow)
        sep_pow_tensor = torch.from_numpy(sep_pow).to(device).transpose(0, 1)

        optimizer = torch.optim.Adam((z, c), lr=1e-3)
        for _ in range(50):
            log_sigma_sq = vae_model.decode(z, torch.softmax(c, dim=1)) + log_g
            loss = torch.sum(log_sigma_sq + (sep_pow_tensor.log() - log_sigma_sq).exp())
            vae_model.zero_grad()
            optimizer.zero_grad()
            loss.backward()  #retain_graph=True
            optimizer.step()

        with torch.no_grad():
            sigma_sq = (vae_model.decode(z, torch.softmax(c, dim=1)) + log_g).exp()
            lbd = torch.sum(sep_pow_tensor / sigma_sq, dim=(1, 2))
            lbd = lbd / n_freq / n_frame / log_g.squeeze(2).squeeze(1).exp()
            log_g[:, 0, 0] += torch.log(lbd)
            sigma_sq *= lbd.unsqueeze(1).unsqueeze(2)
            sep_mat *= lbd.unsqueeze(0).unsqueeze(2).cpu().numpy()
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

    return sep, flag


def mvae_ge2e(mix, vae_model, spkr_model, fb_mat=None, n_iter=600, device=None, convg_tol=1e-6):
    flag = False
    mix = mix.swapaxes(1, 2)
    n_freq, n_src, n_frame = mix.shape
    eye = np.tile(np.eye(n_src), (n_freq, 1, 1))

    sep = mix.copy()
    sep_mat = np.tile(np.eye(n_src), (n_freq, 1, 1)).astype(np.complex64)
    # c = torch.empty(n_src, vae_model.d_embedding).uniform_(-0.1, 0.1).to(device)
    # c = c / torch.norm(c, dim=1, keepdim=True)
    sep_pow = np.power(np.abs(sep), 2)
    log_g = torch.full((n_src, 1, 1), vae_model.log_g.item(), device=device)

    with torch.no_grad():
        sep_pow_tensor = torch.from_numpy(sep_pow).transpose(0, 1).to(device)
        sep_pow_tensor.clamp_(EPS)
        sep_mel_tensor = spectrogram_normalize(sep_pow_tensor)
        sep_mel_tensor = torch.matmul(fb_mat.to(device), sep_mel_tensor)
        sep_mel_tensor = 10 * torch.log10(torch.clamp(sep_mel_tensor, 1e-10))
        c = spkr_model.extract_embd(sep_mel_tensor)
        z, _ = vae_model.encode(sep_pow_tensor, c)
        sigma_sq = (vae_model.decode(z, c) + log_g).exp()
        sigma_sq.clamp_(min=EPS)
        sigma_reci = (1 / sigma_sq).cpu().numpy()
    z.requires_grad = True

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

        optimizer = torch.optim.Adam((z, ), lr=1e-3)
        for _ in range(10):
            with torch.no_grad():
                sep_mel_tensor = spectrogram_normalize(sep_pow_tensor)
                sep_mel_tensor = torch.matmul(fb_mat.to(device), sep_mel_tensor)
                sep_mel_tensor = 10 * torch.log10(torch.clamp(sep_mel_tensor, 1e-10))
                c = spkr_model.extract_embd(sep_mel_tensor)
            log_sigma_sq = vae_model.decode(z, c) + log_g
            loss = torch.sum(log_sigma_sq + (sep_pow_tensor.log() - log_sigma_sq).exp())
            vae_model.zero_grad()
            spkr_model.zero_grad()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        with torch.no_grad():
            sigma_sq = (vae_model.decode(z, c) + log_g).exp()
            lbd = torch.sum(sep_pow_tensor / sigma_sq, dim=(1, 2))
            lbd = lbd / n_freq / n_frame
            log_g[:, 0, 0] += torch.log(lbd)
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
            print('MVAE-ge2e-Iterations: {}, Objective: {}, dObj: {}'.format(t, Obj, dObj))

    sep_mat_inv = np.linalg.inv(sep_mat)
    for n in range(n_src):
        E = np.zeros((n_src, n_src))
        E[n, n] = 1
        tmp = sep_mat_inv @ (E @ sep_mat) @ mix
        sep[:, n, :] = tmp[:, 1, :]

    return sep, flag


if __name__ == "__main__":
    mix_wav_path = '/data/hdd0/zhaoyigu/DATASET/tmp/male_female_aec.wav'
    output_path = '/data/hdd0/zhaoyigu/DATASET/tmp/mvae/sep_mvae_2ch.wav'
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(output_path)
    vae_model_path = '/data/hdd0/zhaoyigu/PROJECT/MVAE_w_embds_data/output/onehot_embds/test_librispeech_500sp_15min/model/state_dict--epoch=2000.pt'
    modelfile_path = 'vae_model_path'
    n_embedding = 500
    device = torch.device(0)
    model = vae.net(n_embedding)
    model.load_state_dict(torch.load(modelfile_path, map_location=device))
    model.to(device)
    model.eval()

    audio, _ = rosa.load(mix_wav_path, sr=16000, mono=False)
    audio = audio[0:2,:]
    # src, _ = rosa.load(src_wav_path, sr=16000, mono=False)
    mix_spec = np.stack([rosa.stft(np.asfortranarray(x_c), n_fft=1024, hop_length=256) for x_c in audio], axis=1)
    sep_spec, _ = myilrma(mix_spec, 1000)
    sep = np.stack([rosa.istft(sep_spec[:, ch, :], hop_length=256) for ch in range(0, src.shape[0])], axis=0)
    sf.write(output_path, sep.T, 16000)
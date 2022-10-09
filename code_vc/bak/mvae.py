def MVAEiva(mix, target_embd, model, n_iter=600, device=None, convg_tol=1e-6,
            ilrma_init=False, alternate_align_init=False, latent_meth='bp', nsamp=None):

    """ Args """
    """ 
    `mix`: [n_freq, n_src, n_frame]
    `target_embd`: [d_embd, ]
    `latent_meth`: 'bp'/'encoder'
    """
    assert len(mix.shape) == 3
    if target_embd.dim() == 1:
        target_embd = target_embd.unsqueeze(0)
    if latent_meth == 'encoder' and nsamp is None:
        print("Only use the mean value of decoder output! Activate sampling by setting `nsamp`!")
        
    
    nan_flag = False
    n_freq, n_src, n_frame = mix.shape
    d_embd = target_embd.shape[1]
    device = target_embd.device
    eye = np.tile(np.eye(n_src), (n_freq, 1, 1))

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
    if target_embd.shape[0] == 1:
        interference_embd = 1 / d_embd * torch.ones(n_src - 1, *target_embd.shape[1:]).to(device)
    elif target_embd.shape[0] == n_src:
        interference_embd = torch.ones(0, *target_embd.shape[1:]).to(device)
    elif target_embd.shape[0] == 0:
        interference_embd = 1 / d_embd * torch.ones(n_src, *target_embd.shape[1:]).to(device)
    log_factor = torch.full((n_src, 1, 1), 1.0, device=device)
    with torch.no_grad():
        spkr_embd = torch.cat((target_embd, interference_embd), dim=0)
        sep_spec_pow_tensor = torch.from_numpy(sep_spec_pow).transpose(0, 1).to(device).float().clamp_min_(EPS)
        content_embd, _ = model.content_encoder(sep_spec_pow_tensor)
        sep_var = (model.decoder(content_embd, spkr_embd) + log_factor).exp().clamp_min_(EPS)[:, :n_freq, :n_frame]
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

        # Update interference_embd
        if latent_meth.startswith('bp'):
            if latent_meth == 'bp_encoderinit':
                with torch.no_grad():
                    content_embd, _ = model.content_encoder(sep_spec_pow_tensor)
                content_embd.requires_grad = True
            optimizer = torch.optim.Adam((content_embd, interference_embd), lr=1e-3)
            for _ in range(10):
                assert interference_embd.requires_grad is True
                assert content_embd.requires_grad is True
                assert target_embd.requires_grad is False
                assert interference_embd.is_leaf is True
                assert content_embd.is_leaf is True
                assert target_embd.is_leaf is True
                spkr_embd = torch.cat((target_embd, interference_embd), dim=0)
                log_sep_var = model.decoder(content_embd, spkr_embd)[:, :n_freq, :n_frame] + log_factor
                loss = torch.sum(log_sep_var + (sep_spec_pow_tensor.log() - log_sep_var).exp())
                model.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        with torch.no_grad():
            sep_var_ori = model.decoder(content_embd, torch.cat((target_embd, interference_embd), dim=0))[:, :n_freq, :n_frame]
            sep_var = (sep_var_ori + log_factor).exp()
            lbd = torch.sum(sep_spec_pow_tensor / sep_var, dim=(1, 2)) / n_freq / n_frame
            log_factor[:, 0, 0] += torch.log(lbd)
            sep_var = (sep_var_ori + log_factor).exp()
            sep_reci = (1 / sep_var).cpu().numpy()

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
        Obj = np.sum(- sep_spec_pow.swapaxes(0, 1) * sep_reci + np.log(sep_reci))\
            + 2 * n_frame * np.sum(np.log(np.abs(np.linalg.det(sep_mat))))
        if np.isnan(Obj):
            nan_flag = True
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
        sep_spec[:, n, :] = tmp[:, 1, :]

    return sep_spec, nan_flag

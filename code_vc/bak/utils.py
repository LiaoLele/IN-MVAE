
class Transform(object):

    def __init__(self, sigproc_param):
        super(Transform, self).__init__()
        self.trans = Spectrogram(**sigproc_param)
    
    def __call__(self, data_target, data_noise, mix_info):
        out = []
        target_spec, _, _ = self.trans(data_target)
        noise_spec, _, _ = self.trans(data_noise)
        for idx, this_mix_info in enumerate(mix_info):
            if this_mix_info is None:
                out.append(target_spec[idx, :, :].clone())
            else:
                noisy_spec = target_spec[idx, :, :].clone()
                this_mix_info = eval(this_mix_info)
                for block in this_mix_info:
                    noisy_spec[block, :] = noise_spec[idx, block, :]
                out.append(noisy_spec)
        out = torch.stack(out, dim=0)
        return target_spec, out

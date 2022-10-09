import os

from config import cfgs
from train import Train

# from config_indev import cfgs
# from train_indev import Train

# from config_indev_new import cfgs
# from train_indev_new import Train

# from config_indev_new_0 import cfgs
# from train_indev_new_0 import Train
from inference import Infer


def ProcessCfgs(cfgs):
    if cfgs.general.train:
        assert cfgs.general.inference is False
        # cfgs.path.model = os.path.join(cfgs.path.root, 'model')
        os.makedirs(cfgs.path.model, exist_ok=True)
        if cfgs.opt_strategy.use_lr_decay:
            assert cfgs.opt_strategy.min_lr is not None
            assert cfgs.opt_strategy.lr_activate_idx is not None
            assert cfgs.opt_strategy.lr_deactivate_idx is not None
        if cfgs.train.nan_backroll:
            assert cfgs.train.max_nan_allowance is not None
    elif cfgs.general.inference:
        assert cfgs.general.train is False


def main(cfgs, seed=0):
    ProcessCfgs(cfgs)

    """ TRAINING """
    if cfgs.general.train:
        assert cfgs.general.inference is False
        Train(cfgs, seed)

    """ INFERENCE """
    if cfgs.general.inference:
        assert cfgs.general.train is False
        Infer(cfgs, seed)


if __name__ == "__main__":
    seed = 0
    main(cfgs, seed=seed)
    



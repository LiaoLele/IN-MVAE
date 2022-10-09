import os
import copy
import pickle


def get_rir_config(out_path, rir_path, rir_path_common_prefix, prefix=None, t60_list=None, angle_list=None):
    """ Get rir_real under the 2-source scenario """
    """ 
    Args:
        `out_path`: dirname of where to save 'idx2rir.txt' and 'key2idx'
        `rir_path`: dirname of MIRD
        `rir_path_common_prefix`: depend on file structure
        `prefix`: depend on which server to use
        `t60_list`: list containing target t60 values
        `angle_list`: if None, use all angle from the dataset; if not None, use provided angle_list
    """
    os.makedirs(out_path, exist_ok=True)
    f_txt = open(os.path.join(out_path, 'idx2rir.txt'), 'w')
    idx2rir = {}
    key2idx = {"key": 't60'}
    idx = 0
    if t60_list is None:
        t60_list = [0.16, 0.36, 0.61]
        t60_list = ['{:.3f}'.format(x) for x in t60_list]
    # if angle_list is None:
    #     angle_list = list(range(0, 90, 15)) + list(range(285, 360, 15))
    #     angle_list = ["{:03d}".format(x) for x in angle_list]
    #     assert len(angle_list) == 11
    if angle_list is None:
        # angle_list = list(range(15, 90, 15))
        angle_list = list(range(30, 90, 15))
    angle_interval = list(range(90, 110, 15))
    dis = '1m'
    mic_interval = "8-8-8-8-8-8-8"
    channel = [3, 4]
    # angle_choice = list(combinations(angle_list, 2))
    for t60 in t60_list:
        rir_final_path = os.path.join(rir_path, "{}_{}_{}".format(rir_path_common_prefix, t60, mic_interval))
        if t60 not in key2idx:
            key2idx[t60] = [[] for _ in range(len(angle_interval))]
        for i, angle_dis in enumerate(angle_interval):
            angle_list_vad = list(filter(lambda x: x + angle_dis <= 150, angle_list))
            # angle_list_vad_copy = copy.deepcopy(angle_list_vad)
            for ang in angle_list_vad:
                angle = [ang, ang + angle_dis]
                # angle_aug = [[ang, ang + angle_dis], [180 - ang, 180 - ang - angle_dis]]
                # angle_aug_copy = copy.deepcopy(angle_aug)
                # for angle in angle_aug_copy:
                angle_copy = copy.deepcopy(angle)
                angle[0] = 90 - angle[0] if angle[0] <= 90 else 450 - angle[0]
                angle[1] = 90 - angle[1] if angle[1] <= 90 else 450 - angle[1]
                angle = ["{:03d}".format(x) for x in angle]
                rir_final_path_1 = list(filter(lambda x: x.endswith("{}s)_{}_{}_{}.mat".format(t60, mic_interval, dis, angle[0])), os.listdir(os.path.join(prefix, rir_final_path))))[0]
                rir_final_path_2 = list(filter(lambda x: x.endswith("{}s)_{}_{}_{}.mat".format(t60, mic_interval, dis, angle[1])), os.listdir(os.path.join(prefix, rir_final_path))))[0]
                key2idx[t60][i].append(idx)
                idx2rir[idx] = {'idx': idx,
                                'path': [os.path.join(rir_final_path, rir_final_path_1), os.path.join(rir_final_path, rir_final_path_2)],
                                'angle': angle,
                                'ori_angle': angle_copy,
                                'channel': channel,
                                'mic_interval': mic_interval,
                                'dis': dis,
                                't60': t60}
                for key, value in idx2rir[idx].items():
                    if key != 'path':
                        print("{}: {}    ".format(key, value), end='', file=f_txt)
                print("\n", file=f_txt)
                idx += 1
    with open(os.path.join(out_path, 'key2idx.pkl'), 'wb') as f:
        pickle.dump(key2idx, f)
    with open(os.path.join(out_path, 'idx2rir.pkl'), 'wb') as f:
        pickle.dump(idx2rir, f)


def main(state=0, prefix=None, sub_state=0):
    if state == 0:
        out_path = os.path.join(prefix, 'DATASET/rir/MIRD')
        rir_path = '/data/hdd0/leleliao/DATASET/rir/'
        rir_path_commom_prefix = 'IR'
        os.system("cp {} {}".format(
            os.path.join(prefix, 'PROJECT/CVAE_training/EsEc_structure/code/PreProcess/RIRGen/get_real_rir.py'),
            os.path.join(out_path, 'get_rir.txt')))
        get_rir_config(out_path, rir_path, rir_path_commom_prefix, prefix='/data/hdd0/leleliao')


if __name__ == "__main__":
    prefix = '/data/hdd0/leleliao'
    main(state=0, sub_state=0, prefix=prefix)

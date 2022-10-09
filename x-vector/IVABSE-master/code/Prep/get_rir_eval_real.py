import random
from itertools import combinations, combinations_with_replacement, product
import os
import pickle
import copy


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
        angle_list = list(range(0, 195, 15))
    angle_interval = list(range(15, 135, 15))
    dis = '1m'
    mic_interval = "8-8-8-8-8-8-8"
    channel = [3, 4]
    # angle_choice = list(combinations(angle_list, 2))
    for t60 in t60_list:
        rir_final_path = os.path.join(rir_path, "{}_{}_{}".format(rir_path_common_prefix, t60, mic_interval))
        if t60 not in key2idx:
            key2idx[t60] = [[] for _ in range(len(angle_interval))]
        for i, angle_dis in enumerate(angle_interval):
            angle_list_vad = list(filter(lambda x: x + angle_dis <= 180, angle_list))
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
        
        
# def get_rir_config(out_path, rir_path, rir_path_common_prefix, prefix=None, t60_list=None, angle_list=None):
#     os.makedirs(out_path, exist_ok=True)
#     f_txt = open(os.path.join(out_path, 'idx2rir.txt'), 'w')
#     idx2rir = {}
#     key2idx = {"key": 't60'}
#     idx = 0
#     t60_list = [0.16, 0.36, 0.61]
#     t60_list = ['{:.3f}'.format(x) for x in t60_list]
#     dis = '1m'
#     mic_interval = "8-8-8-8-8-8-8"
#     channel = [2, 3, 4, 5]
#     angle_list_all = []

#     """ 四个在一侧 """ 
#     angle_list = [[30, 45, 60, 75], [15, 45, 75, 90]]  #固定的
#     angle_list.extend([[180 - x for x in a_list] for a_list in angle_list])
#     angle_list_all = angle_list_all + angle_list
#     """ 三个在一侧 """
#     angle_list = [[15, 45, 75, x] for x in range(105, 190, 15)]
#     angle_list.extend([[180 - x for x in a_list] for a_list in angle_list])
#     angle_list_all = angle_list_all + angle_list
#     """ 两个在一侧 """
#     src_interval_left = list(range(15, 100, 15))
#     src_interval_right = list(range(15, 90, 15))
#     total_combine = list(combinations_with_replacement(range(len(src_interval_right)), 2))
#     total_combine = total_combine + list(product([len(src_interval_left) - 1], range(len(src_interval_right))))
#     angle_left = list(range(0, 100, 15))
#     angle_right = list(range(105, 190, 15))
#     angle_left_comb = []  # [[15], [30], ..., [90]]
#     angle_right_comb = []
#     for interval in src_interval_left:
#         angle_left_vad = list(filter(lambda x: x + interval <= 90, angle_left))
#         angle_left_comb.append([[x, x + interval] for x in angle_left_vad])
#     for interval in src_interval_right:
#         angle_right_vad = list(filter(lambda x: x + interval <= 180, angle_right))
#         angle_right_comb.append([[x, x + interval] for x in angle_right_vad])
#     angle_list = []
#     for idx_left, idx_right in total_combine:
#         random.shuffle(angle_left_comb[idx_left])
#         random.shuffle(angle_right_comb[idx_right])
#         angle_list_tmp = [left_pair + right_pair for left_pair, right_pair in product(angle_left_comb[idx_left], angle_right_comb[idx_right])]
#         random.shuffle(angle_list_tmp)
#         angle_list.extend(random.sample(angle_list_tmp, 2) if len(angle_list_tmp) >= 2 else angle_list_tmp)
#     angle_list_all = angle_list_all + angle_list
#     angle_list = [[90 - x if x <= 90 else 450 - x for x in a_list] for a_list in angle_list_all]

#     for t60 in t60_list:
#         if t60 not in key2idx:
#             key2idx[t60] = []
#         rir_final_path = os.path.join(rir_path, "{}_{}_{}".format(rir_path_common_prefix, t60, mic_interval))
#         angle_list_copy = copy.deepcopy(angle_list)
#         for angle in angle_list_copy:
#             angle = ["{:03d}".format(x) for x in angle]
#             path = [
#                 os.path.join(
#                     rir_final_path, list(filter(lambda x: x.endswith("{}s)_{}_{}_{}.mat".format(t60, mic_interval, dis, angle[i])), os.listdir(os.path.join(prefix, rir_final_path))))[0]
#                 ) for i in range(len(angle))]
#             key2idx[t60].append(idx)
#             idx2rir[idx] = {'idx': idx,
#                             'path': path,
#                             'angle': angle,
#                             'channel': channel,
#                             'mic_interval': mic_interval,
#                             'dis': dis,
#                             't60': t60}
#             for key, value in idx2rir[idx].items():
#                 if key != 'path':
#                     print("{}: {}    ".format(key, value), end='', file=f_txt)
#             print("\n", file=f_txt)
#             idx += 1
            
#     with open(os.path.join(out_path, 'key2idx.pkl'), 'wb') as f:
#         pickle.dump(key2idx, f)
#     with open(os.path.join(out_path, 'idx2rir.pkl'), 'wb') as f:
#         pickle.dump(idx2rir, f)
    

# def get_rir_config(out_path, rir_path, rir_path_common_prefix, prefix=None, t60_list=None, angle_list=None):
#     """ Get rir real for specified source location [usually for more than 3 srcs] """
#     os.makedirs(out_path, exist_ok=True)
#     f_txt = open(os.path.join(out_path, 'idx2rir.txt'), 'w')
#     idx2rir = {}
#     key2idx = {"key": 't60'}
#     idx = 0
#     t60_list = [0.16, 0.36, 0.61]
#     t60_list = ['{:.3f}'.format(x) for x in t60_list]
#     dis = ['1m', '2m']
#     mic_interval = "8-8-8-8-8-8-8"
#     # channel = [2, 3, 4, 5]
#     channel = [3, 4, 5]
#     # angle_list_all = [[30, 75, 120, 150], [30, 90, 120, 165], [15, 45, 75, 150], [30, 45, 105, 150], [30, 45, 105, 120]]
#     angle_list_all = [[90, 30, 150]]  #, [30, 90, 150], [75, 120, 165], [15, 45, 75], [30, 45, 120], [60, 90, 105], [15, 30, 60], [30, 45, 60]]
#     angle_list = [[90 - x if x <= 90 else 450 - x for x in a_list] for a_list in angle_list_all]

#     for t60 in t60_list:
#         rir_final_path = os.path.join(rir_path, "{}_{}_{}".format(rir_path_common_prefix, t60, mic_interval))
#         for i, angle in enumerate(angle_list):
#             key = "{}-{}".format(t60, '_'.join([str(x) for x in angle_list_all[i]]))
#             if key not in key2idx:
#                 key2idx[key] = []
#             angle = ["{:03d}".format(x) for x in angle]
#             path = [
#                 os.path.join(
#                     rir_final_path, list(filter(lambda x: x.endswith("{}s)_{}_{}_{}.mat".format(t60, mic_interval, dis[0] if i == 0 else dis[1], angle[i])), os.listdir(os.path.join(prefix, rir_final_path))))[0]
#                 ) for i in range(len(angle))]
#             key2idx[key].append(idx)
#             idx2rir[idx] = {'idx': idx,
#                             'path': path,
#                             'angle': angle,
#                             'channel': channel,
#                             'mic_interval': mic_interval,
#                             'dis': dis,
#                             't60': t60}
#             for key, value in idx2rir[idx].items():
#                 if key != 'path':
#                     print("{}: {}    ".format(key, value), end='', file=f_txt)
#             print("\n", file=f_txt)
#             idx += 1
            
#     with open(os.path.join(out_path, 'key2idx.pkl'), 'wb') as f:
#         pickle.dump(key2idx, f)
#     with open(os.path.join(out_path, 'idx2rir.pkl'), 'wb') as f:
#         pickle.dump(idx2rir, f)


def main(state=0, prefix=None, sub_state=0):
    if state == 0:
        out_path = os.path.join(prefix, 'DATASET/SEPARATED_LIBRISPEECH/RIR/test_real_remix/')
        rir_path = 'DATASET/MIRD/'
        rir_path_commom_prefix = 'IR'
        os.system("cp {} {}".format(
            os.path.join(prefix, 'DATASET/SEPARATED_LIBRISPEECH/code/Prep/get_rir_eval_real.py'),
            os.path.join(out_path, 'get_rir_eval.txt')))
        get_rir_config(out_path, rir_path, rir_path_commom_prefix, prefix=prefix)


if __name__ == "__main__":
    prefix_other = '/home/user/zhaoyi.gu/mnt/g2'
    prefix_g2 = '/data/hdd0/zhaoyigu'
    main(state=0, sub_state=0, prefix=prefix_g2)

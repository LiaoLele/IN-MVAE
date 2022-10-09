import sys
import random
from itertools import combinations, product
import numpy as np
import pickle
import os
import math
import pyroomacoustics as pra
import soundfile as sf
from scipy.signal import convolve
import copy


def get_room_params(out_path, t60_list, width_range, height_limit, length_max, sabine=True, **determine):
    """ Generate some rooms for each t60_list """
    """ 
    Args:
        `out_path`: [str] path where room_param.pkl/txt will be saved
        `t60_list`: [list] list including t60s-of-interest, 
                           for each t60, len(width_range) numbers of rooms will be created
        `width_range`: [list([list], )] each list in width_range consists of upper and lower limit of width,
                                        for each list in width range, a rooms will be created 
        `height_limit`: [list/tuple] miminum and maximum value for height
        'length_max': [float] maximum length of the long side of the room
        `sabin`: [bool] whether the room is sabine room i.e. max-dim-len/min-dim-len<3
    
    Out:
        `ret`: [dict][not returned but saved] 
               {t60-0: [(longside, shortside, height, absorption), (param-for-room2), ...], t60-1: [(), (), ...], ...} 
        `room_param.pkl`: pickle file that saves ret
        `room_param.txt`: txt file that writes each room parameters into a single line
    """

    ret = {}
    f_txt = open(os.path.join(out_path, 'room_param.txt'), 'wt')
    if f_txt is None:
        sys.exit('Error opening file: ' + os.path.join(out_path, 'room_param.txt'))
    f_pkl = open(os.path.join(out_path, 'room_param.pkl'), 'wb')
    if f_pkl is None:
        sys.exit('Error opening file: ' + os.path.join(out_path, 'room_param.pkl'))

    if not determine:
        """ 随机生成 """
        for t60 in t60_list:
            if t60 not in ret:
                ret[t60] = []
            # generate rooms for each t60 
            # 考虑因素：短边从width_range的每个list标定的范围内均匀分布选取;
            #          长边不小于短边，不大于length_max;
            #          高度从height_limit标定的范围内均匀分布选取   
            for width in width_range:
                ww = round(random.uniform(width[0], width[1]), 2)
                hh = round(random.uniform(height_limit[0], height_limit[1]), 2)
                ll = round(random.uniform(ww, min(3 * ww, length_max)), 2) if sabine else round(random.uniform(ww, length_max), 2)
                V = ww * hh * ll
                S = 2 * np.sum([np.prod(x) for x in combinations([ll, ww, hh], 2)])
                alpha = 1 - np.exp(-0.161 * V / (S * t60))
                ret[t60].append((ll, ww, hh, alpha))
                print('t60: {:.2f}, width: {:.2f}, length: {:.2f}, height: {:.2f}, alpha: {:.2f}'.format(t60, ww, ll, hh, alpha), file=f_txt)
    else:
        """ 根据给定的房间大小生成 """
        for t60, room_list in determine.items():
            t60 = float(t60)
            if t60 not in ret:
                ret[t60] = []
            for room in room_list:
                ll, ww, hh = room
                V = ww * hh * ll
                S = 2 * np.sum([np.prod(x) for x in combinations([ll, ww, hh], 2)])
                alpha = 1 - np.exp(-0.161 * V / (S * t60))
                ret[t60].append((ll, ww, hh, alpha))
                print('t60: {:.2f}, width: {:.2f}, length: {:.2f}, height: {:.2f}, alpha: {:.2f}'.format(t60, ww, ll, hh, alpha), file=f_txt)
            
    pickle.dump(ret, f_pkl)
    f_txt.close()
    f_pkl.close()


def get_mic_src_params(out_path, mic_interval_limit, src_distance_min, src_angle_range,
                       src_angle_diff_min=20, num_conf_per_room=1, src_wall_min=0.5,
                       **determine):
    """ Generate some rir for each room """
    """ 
    Args:
        `out_path`: [str] path where room_parm.pkl is saved and rir_param.pkl/txt will be saved
        `mic_interval_limit`: [2-element tuple/list in order] minimum and maximum mic_interval
        `src_distance_min`: [float] minimum distance between src and mic center
        `src_angle_range`: [list] valid src angle values
        `src_angle_diff_min`: [float/int] minimum angle differences between two srcs
        `num_conf_per_room`: [int] number of rirs generated for each room configuration
        `src_wall_min`: [float] minimum vertical distance between src and walls

    Out:
        `rir_param`: [dict][not returned but saved]
                      {t60-0: [{params-for-rir-0}, {params-for-rir-1}, ...], t60-2: [{}, {}, ...], ... }
        `rir_param.pkl`: pickle file that saves rir_param
        `rir_param.txt`: txt file that writes each rir into a single line
    """

    rir_param = {}
    f_room_conf = open(os.path.join(out_path, 'room_param.pkl'), 'rb')
    if f_room_conf is None:
        sys.exit('Error opening file: ' + os.path.join(out_path, 'room_param.pkl'))
    f_rir_conf = open(os.path.join(out_path, 'rir_param.pkl'), 'wb')
    if f_rir_conf is None:
        sys.exit('Error opening file: ' + os.path.join(out_path, 'rir_param.pkl'))
    f_rir_conf_txt = open(os.path.join(out_path, 'rir_param_txt.txt'), 'wt')
    if f_rir_conf_txt is None:
        sys.exit('Error opening file: ' + os.path.join(out_path, 'rir_param_txt.pkl'))
    room_conf = pickle.load(f_room_conf)
    f_room_conf.close()
    global_idx = 0  

    angle2idx = {}
    for t60 in room_conf:
        local_idx = 0   
        rir_param[t60] = []
        for ll, ww, hh, alpha in room_conf[t60]:
            """ 随机生成 """
            if not determine:
                # Determine rir config for this room
                # 考虑因素：每个房间都生成num_conf_per_room个rir
                cnt = 0
                S = 2 * np.sum([np.prod(x) for x in combinations([ll, ww, hh], 2)])
                R = S * alpha / (1 - alpha)          
                critical_dis = 0.25 * math.sqrt(R / math.pi)
                while cnt < num_conf_per_room:
                    # Determine mic center and angle[degree] and interval
                    # 考虑因素：若边长大于3 m，麦克风距该边的距离不得小于1.5 m，否则不得小于1 m
                    #          麦克风高度为1.2 m
                    #          麦克风的选择角度用标准极坐标系中的相交表示
                    #          麦克风的间距在 mic_interval_limit标定的最小值和最大值之间均匀分布选取
                    mic_c = [random.uniform(1.5, ll - 1.5) if ll > 3 else random.uniform(1.0, ll - 1.0), 
                             random.uniform(1.5, ww - 1.5) if ww > 3 else random.uniform(1.0, ww - 1.0),
                             1.2]
                    mic_angle = random.randrange(0, 360)
                    mic_interval = round(random.uniform(mic_interval_limit[0], mic_interval_limit[1]), 2)
                    nmic = 2

                    # Determine src angle[degree] in abs_src_angle
                    # 考虑因素：两声源在麦克风阵列同侧；
                    #          声源相对于阵列的角度从src_angle_range中选择
                    #          两声源的角度差不得小于src_angle_diff_min度
                    #          声源最终的角度用标准极坐标系中的相角表示
                    src_angle_1 = random.choice(src_angle_range)
                    if src_angle_1 <= 180:
                        src_angle_range_2 = list(filter(lambda x: x <= 180 and abs(x - src_angle_1) >= src_angle_diff_min, src_angle_range))
                        src_angle_2 = random.choice(src_angle_range_2)
                    else:
                        src_angle_range_2 = list(filter(lambda x: x >= 180 and abs(x - src_angle_1) >= src_angle_diff_min, src_angle_range))
                        src_angle_2 = random.choice(src_angle_range_2)
                    abs_src_angle_1 = mic_angle + src_angle_1
                    abs_src_angle_2 = mic_angle + src_angle_2
                    abs_src_angle = [abs_src_angle_1, abs_src_angle_2]
                    abs_src_angle = list(map(lambda x: x if x < 360 else x - 360, abs_src_angle))

                    # calculate the max distance of source limited by abs_src_angle in src_dis_limby_angle 
                    # 分象限讨论边界约束
                    # 考虑因素：声源离墙垂直距离不得小于src_wall_min
                    src_dis_limby_angle = []
                    for src_angle in abs_src_angle:
                        assert src_angle >= 0 and src_angle < 360
                        if src_angle == 0:
                            dis_limby_angle = ll - src_wall_min - mic_c[0]
                        elif src_angle > 0 and src_angle < 90:
                            dis_limby_angle = min((ll - src_wall_min - mic_c[0]) / math.cos(src_angle * math.pi / 180),
                                                  (ww - src_wall_min - mic_c[1]) / math.sin(src_angle * math.pi / 180)) 
                        elif src_angle == 90:
                            dis_limby_angle = ww - src_wall_min - mic_c[1]
                        elif src_angle > 90 and src_angle < 180:
                            dis_limby_angle = min(-(mic_c[0] - src_wall_min) / math.cos(src_angle * math.pi / 180),
                                                  (ww - src_wall_min - mic_c[1]) / math.sin(src_angle * math.pi / 180)) 
                        elif src_angle == 180:
                            dis_limby_angle = mic_c[0] - src_wall_min
                        elif src_angle > 180 and src_angle < 270:
                            dis_limby_angle = min(-(mic_c[0] - src_wall_min) / math.cos(src_angle * math.pi / 180),
                                                  -(mic_c[1] - src_wall_min) / math.sin(src_angle * math.pi / 180)) 
                        elif src_angle == 270:
                            dis_limby_angle = mic_c[1] - src_wall_min
                        elif src_angle > 270 and src_angle < 360:
                            dis_limby_angle = min((ll - src_wall_min - mic_c[0]) / math.cos(src_angle * math.pi / 180),
                                                  -(mic_c[1] - src_wall_min) / math.sin(src_angle * math.pi / 180))
                        else:
                            raise ValueError('src_angle not in the correct range')
                        src_dis_limby_angle.append(dis_limby_angle)

                    # Determine src distance in src_distance
                    # 考虑因素：声源距麦克风至少大于0.5 m，不得超出临界距离外0.5 m；
                    #          两声源和麦克风间距的差值不应大于0.2 m
                    #          声源距离墙面垂直距离不得小于src_wall_min
                    #          声源高度固定为1.2 m
                    src_dis_1 = random.uniform(src_distance_min, min(critical_dis + 0.5, src_dis_limby_angle[0]))
                    src_dis_min_2 = max(src_distance_min, src_dis_1 - 0.5)
                    src_dis_max_2 = min(critical_dis + 0.5, src_dis_1 + 0.5, src_dis_limby_angle[1])
                    if src_dis_max_2 < src_dis_min_2:
                        continue
                    src_dis_2 = random.uniform(src_dis_min_2, src_dis_max_2)
                    src_distance = [src_dis_1, src_dis_2]
                    src_height = [1.2, 1.2]

                    # Determine src parameters in src_setting [np.ndarray]
                    src_setting = np.stack((np.array(abs_src_angle), np.array(src_distance), np.array(src_height)), axis=1)

                    # Write params needed to compute rir
                    param_dict = {'global_idx': global_idx, 'local_idx': local_idx,
                                  'room_size': [ll, ww, hh], 'absorption': alpha,
                                  'nmic': nmic, 'mic_center': np.array(mic_c), 
                                  'mic_interval': mic_interval, 'mic_angle': mic_angle,
                                  'src_setting': src_setting, 'relative_src_angle': [src_angle_1, src_angle_2]}
                    rir_param[t60].append(param_dict)
                    for key, value in param_dict.items():
                        print('{}: {}    '.format(key, value), end='', file=f_rir_conf_txt)
                    print('\n', file=f_rir_conf_txt)

                    cnt += 1
                    global_idx += 1   # 所有rir中的index，从0开始
                    local_idx += 1   # 每种t60内部的index，从0开始
            else:
                """ 根据给定的设置生成 """
                for nmic, mic_c, mic_interval, mic_angle, src_setting in product(determine['mic_num'], determine['mic_center'], determine['mic_interval'], determine['mic_angle'], determine['src_setting']):
                    if determine['src_angle_interval'] is None:
                        """ 固定的角度 """
                        src_setting_copy = copy.deepcopy(src_setting)
                        src_setting_copy_2 = copy.deepcopy(src_setting)
                        for i in range(len(src_setting_copy[0])):
                            src_setting_copy[0][i] += mic_angle
                            if src_setting_copy[0][i] > 360:
                                src_setting_copy[0][i] -= 360
                        src_setting_np = np.stack((np.array(src_setting_copy[0]), np.array(src_setting_copy[1]), np.array(src_setting_copy[2])), axis=1)
                        param_dict = {'global_idx': global_idx, 'local_idx': local_idx,
                                      'room_size': [ll, ww, hh], 'absorption': alpha,
                                      'nmic': nmic, 'mic_center': np.array(mic_c), 
                                      'mic_interval': mic_interval, 'mic_angle': mic_angle,
                                      'src_setting': src_setting_np, 'relative_src_angle': [src_setting_copy_2[0][0], src_setting_copy_2[0][1]]}
                        rir_param[t60].append(param_dict)
                        for key, value in param_dict.items():
                            print('{}: {}    '.format(key, value), end='', file=f_rir_conf_txt)
                        print('\n', file=f_rir_conf_txt)

                        global_idx += 1   # 所有rir中的index，从0开始
                        local_idx += 1   # 每种t60内部的index，从0开始
                    else:
                        """ 随机生成的角度 """
                        for angle_interval in determine['src_angle_interval']:
                            valid_angle_range = list(filter(lambda x: (x + angle_interval) <= 170, determine['angle_range']))
                            if angle_interval not in angle2idx:
                                angle2idx["{}-{}".format(t60, angle_interval)] = []
                            for angle in valid_angle_range:
                                src_setting[0] = []
                                src_setting[0].append(angle)
                                src_setting[0].append(angle + angle_interval)

                                src_setting_copy = copy.deepcopy(src_setting)
                                for i in range(len(src_setting[0])):
                                    src_setting[0][i] += mic_angle
                                    if src_setting[0][i] > 360:
                                        src_setting[0][i] -= 360
                                src_setting_np = np.stack((np.array(src_setting[0]), np.array(src_setting[1]), np.array(src_setting[2])), axis=1)
                                param_dict = {'global_idx': global_idx, 'local_idx': local_idx,
                                              'room_size': [ll, ww, hh], 'absorption': alpha,
                                              'nmic': nmic, 'mic_center': np.array(mic_c), 
                                              'mic_interval': mic_interval, 'mic_angle': mic_angle,
                                              'src_setting': src_setting_np, 'relative_src_angle': [src_setting_copy[0][0], src_setting_copy[0][1]]}
                                rir_param[t60].append(param_dict)
                                for key, value in param_dict.items():
                                    print('{}: {}    '.format(key, value), end='', file=f_rir_conf_txt)
                                print('\n', file=f_rir_conf_txt)

                                angle2idx["{}-{}".format(t60, angle_interval)].append(global_idx)
                                global_idx += 1   # 所有rir中的index，从0开始
                                local_idx += 1   # 每种t60内部的index，从0开始
                    
    pickle.dump(rir_param, f_rir_conf)
    f_rir_conf.close()
    f_rir_conf_txt.close()
    try:
        if angle2idx:
            f = open(os.path.join(out_path, 'angle2gidx.pkl'), 'wb')
            pickle.dump(angle2idx, f)
            f.close()
    except NameError:
        pass

                    
def get_rir_data(out_path, fs=16000):
    """ Generate rir data for each rir_param """
    """
    Args:
        `out_path`: path where rir_param.pkl is saved and t602gidx.pkl and gidx2rir will be saved

    Out:
        `t602gidx`: [dict][not returned but saved]
                    {t60-0: [list of global indices], t60-1: [list of global indices], ...}
        `idx2rir`: [dict][Not returned but saved]
                    {global-index-0: (local-index, rir-0), global-index-1: (local-idx, rir-1), ...}
        `t602gidx.pkl`: pickle file that saves t602gidx
        `idx2rir.pkl`: pickle file that saves idx2rir
    """

    f_rir_conf = open(os.path.join(out_path, 'rir_param.pkl'), 'rb')
    if f_rir_conf is None:
        sys.exit('Error opening file: ' + os.path.join(out_path, 'rir_param.pkl'))
    rir_param = pickle.load(f_rir_conf)
    f_rir_conf.close()

    f_t602gidx = open(os.path.join(out_path, 't602gidx.pkl'), 'wb')
    if f_t602gidx is None:
        sys.exit('Error opening file: ' + os.path.join(out_path, 't602gidx.pkl'))
    f_idx2rir = open(os.path.join(out_path, 'idx2rir.pkl'), 'wb')
    if f_idx2rir is None:
        sys.exit('Error opening file: ' + os.path.join(out_path, 'idx2rir.pkl'))
    t602gidx = {}
    idx2rir = {}

    for t60 in rir_param:
        if t60 not in t602gidx:
            t602gidx[t60] = []

        for rir_conf in rir_param[t60]:
            room_size = rir_conf['room_size']
            absorption = rir_conf['absorption']
            nmic = rir_conf['nmic']
            mic_center = rir_conf['mic_center']
            mic_interval = rir_conf['mic_interval']
            mic_angle = rir_conf['mic_angle']
            src_setting = rir_conf['src_setting']

            mic_angle = mic_angle * math.pi / 180
            src_setting[:, 0] = src_setting[:, 0] * math.pi / 180

            # create room
            room = pra.ShoeBox(room_size, fs=fs, absorption=absorption, max_order=32)
            # add mic
            relative_mic_pos = np.linspace(-mic_interval * (nmic - 1) / 2, mic_interval * (nmic - 1) / 2, nmic)
            assert relative_mic_pos.shape[0] == nmic
            mic_coordinates = np.zeros((3, nmic)) 
            for i, mic_pos in enumerate(relative_mic_pos):
                mic_coordinates[0, i] = mic_center[0] + mic_pos * math.cos(mic_angle)
                mic_coordinates[1, i] = mic_center[1] + mic_pos * math.sin(mic_angle)
                mic_coordinates[2, i] = mic_center[2]
            mic_array = pra.MicrophoneArray(mic_coordinates, room.fs)
            room.add_microphone_array(mic_array)
            # add src 
            for src in src_setting:
                src_coordinate = [0.0, 0.0, src[2]]
                src_coordinate[0] = mic_center[0] + src[1] * math.cos(src[0])
                src_coordinate[1] = mic_center[1] + src[1] * math.sin(src[0]) 
                room.add_source(src_coordinate)
            # compute rir
            room.compute_rir()
            ret = room.rir
            min_len = min([len(ret[m][n]) for m, n in product(range(len(ret)), range(len(ret[0])))])
            for m, s in product(range(len(ret)), range(len(ret[0]))):
                ret[m][s] = ret[m][s][:min_len]
            
            t602gidx[t60].append(rir_conf['global_idx'])
            idx2rir[rir_conf['global_idx']] = (rir_conf['local_idx'], ret)

    pickle.dump(t602gidx, f_t602gidx)
    pickle.dump(idx2rir, f_idx2rir)
    f_t602gidx.close()
    f_idx2rir.close()


def main(state=0):
    out_path = '/data/hdd0/zhaoyigu/DATASET/SEPARATED_LIBRISPEECH/RIR/test_clean/t60_angle_interval_study'
    os.makedirs(out_path, exist_ok=True)

    if state == 0: 
        """ Generate Room Configurations room_conf.pkl """
        t60_list = list(np.linspace(0.1, 0.7, 7))
        width_range = [[3, 5], [5, 10], [10, 15]] * 7
        height_limit = [2, 4]
        length_max = 15
        sabine = True

        determine = {'0.15': [[8, 7, 3]], '0.25': [[8, 7, 3]], '0.35': [[8, 7, 3]], '0.45': [[8, 7, 3]], '0.55': [[8, 7, 3]], '0.65': [[8, 7, 3]]}

        get_room_params(out_path, t60_list, width_range, height_limit, length_max, sabine=sabine, **determine)

    elif state == 1:
        """ Generate RIR Configurations rir_param.pkl """
        mic_interval_limit = [0.1, 0.3]
        src_distance_min = 0.5
        src_angle_range = list(range(10, 180, 10)) + list(range(190, 360, 10))
        src_angle_diff_min = 20
        num_conf_per_room = 30
        src_wall_min = 0.5

        src_setting = [[[], [1.0, 1.0], [1.2, 1.2]]]
        determine = {'mic_num': [2],
                     'mic_center': [[4.8, 4.3, 1.2]],
                     'mic_interval': [0.1],
                     'mic_angle': [90],
                     'src_setting': src_setting,
                     'src_angle_interval': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
                     'angle_range': list(range(10, 90, 10)),
                     }

        get_mic_src_params(out_path, mic_interval_limit, src_distance_min, src_angle_range,
                           src_angle_diff_min=src_angle_diff_min, num_conf_per_room=num_conf_per_room, src_wall_min=src_wall_min,
                           **determine)

    elif state == 2:
        """ Generate RIRs t602gidx.pkl and idx2rir.pkl """
        get_rir_data(out_path, fs=16000)

    elif state == 3:
        f_t602gidx = open(os.path.join(out_path, 't602gidx.pkl'), 'rb')
        f_idx2rir = open(os.path.join(out_path, 'idx2rir.pkl'), 'rb')
        t602gidx = pickle.load(f_t602gidx)
        idx2rir = pickle.load(f_idx2rir)
        f_t602gidx.close()
        f_idx2rir.close()

        test_wav_path = '/data/hdd0/zhaoyigu/PROJECT/MVAE_speakerencode_data/mixture/librispeech_test_dev_clean_0211/t60_0.55/'
        test_out_path = '/data/hdd0/zhaoyigu/PROJECT/MVAE_speakerencode_data/mixture/librispeech_test_dev_clean_0211/test/'
        os.makedirs(test_out_path, exist_ok=True)
        src_list = list(filter(lambda x: x.endswith('src.wav'), os.listdir(test_wav_path)))
        j = 0
        for i in range(5):
            for t60 in t602gidx:
                rir_idx = random.choice(t602gidx[t60])
                rir = idx2rir[rir_idx][1]
                src, _ = sf.read(os.path.join(test_wav_path, src_list[j]))
                mix_1 = convolve(src[:, 0], rir[0][0]) + convolve(src[:, 1], rir[0][1])
                mix_2 = convolve(src[:, 0], rir[1][0]) + convolve(src[:, 1], rir[1][1])
                mix = np.stack((mix_1, mix_2), axis=1)
                sf.write(os.path.join(test_out_path, src_list[j][0:-4] + '{}-{}-{}.wav'.format(t60, i, rir_idx)), mix[0: 80000,:], 16000)
                j += 1

 
if __name__ == "__main__":
    main(state=2) 
import pickle
import os
from collections import defaultdict

def CatInfoFile(infofile_list):
    pass


def Get_Egs(infofile, out_dir, stft_hop, frame_num, utter_hop=None, random=False):
    f = open(infofile, 'rb')
    info = pickle.load(f)
    f.close()

    f_curl = open(os.path.join(out_dir, 'egs_train.pkl'), 'wb')
    # f_spread = open(os.path.join(out_dir, 'DataIdxSprd.pkl'), 'wb')
    info_curl = defaultdict(int)
    # info_spread = []

    utter_len = int((frame_num - 1) * stft_hop)
    utter_hop = int(utter_len * utter_hop)

    for idx, line in enumerate(info):
        # info_curl.append([])
        info_curl_tuple = []
        max_num = int((line[1] - utter_len) // utter_hop + 1)
        for i in range(max_num):
            info_curl_tuple.append((line[0], i * utter_hop, utter_len))
            # info_curl[idx].append((line[0], i * utter_hop, utter_len))
            # info_curl[idx].append((line[0], line[2], i * utter_hop, i * utter_hop + utter_len))
            # info_spread.append((line[0], line[2], i * utter_hop, i * utter_hop + utter_len))
        info_curl[idx]=info_curl_tuple
    info_curl = [info_curl]
    pickle.dump(info_curl, f_curl)
    # pickle.dump(info_spread, f_spread)

    f_curl.close()
    # f_spread.close()


if __name__ == "__main__":
    infofile = '/data/hdd0/zhaoyigu/DATASET/Librispeech/concatenate/train_clean/info.pkl'
    out_dir = '/home/nis/lele.liao/projects/code_vc/PLDA/data/'
    stft_hop = 256
    frame_num = 1876
    utter_hop = 1
    Get_Egs(infofile, out_dir, stft_hop, frame_num, utter_hop=utter_hop)









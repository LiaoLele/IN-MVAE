import os
import numpy as np
import copy
import pickle
import scipy.io as sio


def main(state=0, prefix=None):
    if state == 0:
        suffix = 'confirm'
        sep_method = ['ILRMA', 'MVAE_onehot_ilrmainit']  # , 'MVAE_onehot'
        datafile_list = [
            # os.path.join(prefix, 'DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu_remix/t60_angle_interval_study_withsir/rkg_hist_ret--{}--SDR.'.format(m) + suffix + '.pkl')
            # os.path.join(prefix, 'DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/rkg_hist_ret--{}--SDR.'.format(m) + suffix + '.pkl')
            os.path.join(prefix, 'DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/extracted_hist_ret--{}--SDR.'.format(m) + suffix + '.pkl')
            for m in sep_method
        ]
        # out_path = os.path.join(prefix, 'DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu_remix/t60_angle_interval_study_withsir/xvec_accuracy_hist.' + suffix + '.mat')
        # out_path = os.path.join(prefix, 'DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/xvec_accuracy_hist.' + suffix + '.mat')
        out_path = os.path.join(prefix, 'DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_real_remix/t60_study_withsir/extracted_SDR_hist.' + suffix + '.mat')
        rkg_method = [
            'xvec_sepaug-plda_withaug_onlyln',
            'xvec_diraug-plda_withaug_onlyln',
            'xvec_withoutaug-plda_withoutaug_onlyln',
        ]
        mat = []
        for file in datafile_list:
            with open(file, 'rb') as f:
                hist = pickle.load(f)
            for rkg in rkg_method:
                # mat.append(hist[rkg][-1])
                mat.append(hist[rkg])
        mat = np.stack(mat, axis=0)
        # sio.savemat(out_path, {'accuracy_hist': mat})
        sio.savemat(out_path, {'SDR_hist': mat})
            

if __name__ == "__main__":
    prefix = ['/data/hdd0/zhaoyigu', '/home/user/zhaoyi.gu/mnt/g2'][0]
    main(state=0, prefix=prefix)

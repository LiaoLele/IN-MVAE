import os
from itertools import product

prefix = "/home/user/zhaoyi.gu/mnt/g2/"
t60 = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
angle_interval = [20, 30, 40, 70, 90, 110]
datafile_list = os.path.join(prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study_withsir/")
orifile_list = os.path.join(prefix, "DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study/")
for t, angle in product(t60, angle_interval):
    name = "{}-{}".format(t, angle)
    os.system("cp -rn {} {}".format(os.path.join(orifile_list, name), datafile_list))
    os.system("rm -r {}".format(os.path.join(datafile_list, name, "GMM")))
    os.system("rm -r {}".format(os.path.join(datafile_list, name, "MVAE_ge2e")))
    os.system("rm -r {}".format(os.path.join(datafile_list, name, "ILRMA", "rkg*")))
    os.system("rm -r {}".format(os.path.join(datafile_list, name, "MVAE_onehot", "rkg*")))

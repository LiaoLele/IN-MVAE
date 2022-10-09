import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.io as sio
import matplotlib.style
import matplotlib as mpl
import os
import copy


def plot_line(data, **kwargs):
    xlabel = kwargs['xlabel']
    ylabel = kwargs['ylabel']
    xticklabel = kwargs['xticklabel']
    yticklabel = kwargs['yticklabel']


def plot_bar(data, out_path, **kwargs):
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    title = kwargs['title']
    xlabel = kwargs['xlabel']
    ylabel = kwargs['ylabel']
    xticklabel = kwargs['xticklabel']
    ylim = kwargs['ylim']
    yticks = kwargs['yticks']
    legend = kwargs['legend']
    dis = kwargs['dis']
    legend_pos = kwargs['legend_pos']

    num = len(data)
    width = (dis - 0.4) / num
    x = np.arange(len(xticklabel)) * dis
    loc = - (num - 1) / 2 * width

    fig, ax = plt.subplots()
    for i in range(len(data)):
        rects = ax.bar(x + loc, data[i], width, label=legend[i])
        loc += width
        # autolabel(rects)

    # Add some text for xticklabel, title and custom x-axis tick labels, etc.
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabel)
    if legend_pos is not None:
        ax.legend(loc=legend_pos)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    if yticks is not None:
        ax.set_yticks(yticks)
    # fig.tight_layout()
    # plt.show()
    fig.savefig(out_path, dpi=600)


def plot_bar_line(data, out_path, **kwargs):
    
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['font.size'] = '15'
    # plt.rcParams['figure.figsize'] = (12.0,5.0) 
    color = ['navy', 'royalblue', 'skyblue']
    marker = ['o', 's', '^']

    # 取做图数据
    bar_data = data['bar_data']    # [[data-for-label-0], [data-for-label-1], ...]
    line_data = data['line_data']  # [[data-for-label-0], [data-for-label-1], ...]
    num_bar = len(bar_data)
    width = (kwargs['dis'] - 0.4) / num_bar
    x = np.arange(len(kwargs['xticklabel'])) * kwargs['dis']
    loc = - (num_bar - 1) / 2 * width

    fig = plt.figure()
    # 画柱子
    ax1 = fig.add_subplot(111)
    for i in range(len(bar_data)):
        ax1.bar(x + loc, bar_data[i], width, label=kwargs['legend'][i], color=[color[i]] * len(bar_data))
        loc += width
    ax1.set_ylabel(kwargs['ylabel_bar'], fontsize='15')
    if 'ylim_bar' in kwargs:
        ax1.set_ylim(bottom=kwargs['ylim_bar'][0], top=kwargs['ylim_bar'][1])
    if 'yticks_bar' in kwargs:
        ax1.set_yticks(kwargs['yticks_bar'])

    # 画折线图
    ax2 = ax1.twinx()  # 这个很重要噢
    for i in range(len(line_data)):
        ax2.plot(x, line_data[i], color=color[i], marker=marker[i], ms=3, label=kwargs['legend'][i], linewidth=1)
    ax2.set_ylabel(kwargs['ylabel_line'], fontsize='15')
    if 'ylim_line' in kwargs:
        ax2.set_ylim(bottom=kwargs['ylim_line'][0], top=kwargs['ylim_line'][1])
    if 'yticks_line' in kwargs:
        ax2.set_yticks(kwargs['yticks_line'])

    # 纵轴标签
    plt.yticks(fontsize=15)
    plt.xticks(x, kwargs['xticklabel'], fontsize=15)
    plt.xlabel(kwargs['xlabel'], fontsize=15)
    plt.grid(False)
    # ax1.set_title("近年同期公司累计保费收入与同比增速", fontsize='20')  
    
    #添加数据标签
    # for x, y ,z in zip(x,y2,y1):
    #         plt.text(x, y+0.3, str(y), ha='center', va='bottom', fontsize=20,rotation=0)
    #         plt.text(x, z-z, str(int(z)), ha='center', va='bottom', fontsize=21,rotation=0)
        
    #保存图片 dpi为图像分辨率
    plt.legend(loc=9, ncol=3, bbox_to_anchor=(0.5, 1.15), fontsize=9)
    plt.show()
    # plt.savefig('e:/tj/month/fx1806/公司保费增速与同比.png',dpi=600,bbox_inches = 'tight')


def np2mat(array, name, out_path):
    mat_dict = {}
    for i in len(name):
        mat_dict[name[i]] = array[i]
    sio.savemat(out_path, mat_dict)


def extract_sepdata_fromfile(datafile_path, key="t60", other=None):
    """ 
    data_dict: ret['0.15-20']['makemix_same_00']["SDR"] = [avg_improvement, total_num_of_output_channe(twice the number of mixtures)]
    """
    def concate_ret(ret, e):
        environ = e[0] + '-' + e[1]
        sdr_all = 0.0
        sir_all = 0.0
        total_num = 0
        for dataname in ret[environ].keys():
            total_num += ret[environ][dataname]["SDR"][1]
            sdr_all += (ret[environ][dataname]["SDR"][0] * ret[environ][dataname]["SDR"][1])
            sir_all += (ret[environ][dataname]["SIR"][0] * ret[environ][dataname]["SDR"][1])
        return sdr_all / total_num, sir_all / total_num

    out = {"SIR": [[] for _ in range(len(datafile_path))], "SDR": [[] for _ in range(len(datafile_path))]}
    if key == 't60':
        label = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
    elif key.startswith('angle'): 
        label = [20, 30, 40, 70, 90, 110]
    for idx, file in enumerate(datafile_path):
        with open(file, 'rb') as f:
            ret = pickle.load(f)
        env = list(ret.keys())
        env = [x.split('-') for x in env]
        for tag in label:
            sdr = 0.0
            sir = 0.0
            num = 0
            for e in env:
                if e[0 if key == 't60' else 1] == str(tag):
                    if other is not None:
                        if e[1 if key == 't60' else 0] != other:
                            continue
                    sdr_part, sir_part = concate_ret(ret, e)
                    sdr += sdr_part
                    sir += sir_part
                    num += 1
            sdr = sdr / num
            sir = sir / num
            out["SDR"][idx].append(sdr)
            out["SIR"][idx].append(sir)
    return out


def concate_env_in_rkg(ret, key="t60", other=None):
    """ 
    data_dict: ret['0.15-20']['makemix_same_00']["SDR"] = [avg_improvement, total_num_of_output_channe(twice the number of mixtures)]
    """
    out = {}
    if key == 't60':
        label = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
    elif key.startswith('angle'): 
        label = [20, 30, 40, 70, 90, 110]
        
    for rkg_meth in ret.keys():
        out[rkg_meth] = {}
        for sep_meth in ret[rkg_meth].keys():
            out[rkg_meth][sep_meth] = {}
            env = list(ret[rkg_meth][sep_meth])
            env = [x.split('-') for x in env]
            for tag in label:
                correct_num = 0
                total_num = 0
                for e in env:
                    if e[0 if key == 't60' else 1] == str(tag):
                        # if other is not None:
                        #     if e[1 if key == 't60' else 0] != other:
                        #         continue
                        correct_part, total_part = ret[rkg_meth][sep_meth][e[0] + '-' + e[1]]
                        correct_num += correct_part
                        total_num += total_part
                out[rkg_meth][sep_meth][tag] = [correct_num, total_num, correct_num / total_num]
    return out


def concate_env_in_sep(ret, key="t60", other=None):
    """ 
    data_dict: ret['0.15-20']['makemix_same_00']["SDR"] = [avg_improvement, total_num_of_output_channe(twice the number of mixtures)]
    """
    out = {}
    if key == 't60':
        label = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
    elif key.startswith('angle'): 
        label = [20, 30, 40, 70, 90, 110]
        
    for rkg_meth in ret.keys():
        out[rkg_meth] = {}
        for sep_meth in ret[rkg_meth].keys():
            out[rkg_meth][sep_meth] = {}
            env = list(ret[rkg_meth][sep_meth])
            env = [x.split('-') for x in env]
            for tag in label:
                out[rkg_meth][sep_meth][tag] = {'SIR': [], 'SDR': []}
                sir_all = 0
                sdr_all = 0
                sir_imp_all = 0.0
                sdr_imp_all = 0.0
                sdr_all_correct = 0.0
                sir_all_correct = 0.0
                sdr_imp_all_correct = 0.0
                sir_imp_all_correct = 0.0
                correct_num = 0
                total_num = 0
                for e in env:
                    if e[0 if key == 't60' else 1] == str(tag):
                        if other is not None:
                            if e[1 if key == 't60' else 0] != other:
                                continue
                        [sir_part_correct, sir_imp_part_correct, correct_part], [sir_part, sir_imp_part, total_part] = ret[rkg_meth][sep_meth][e[0] + '-' + e[1]]['SIR']
                        [sdr_part_correct, sdr_imp_part_correct, correct_part], [sdr_part, sdr_imp_part, total_part] = ret[rkg_meth][sep_meth][e[0] + '-' + e[1]]['SDR']
                        sir_all += sir_part * total_part
                        sdr_all += sdr_part * total_part
                        sir_imp_all += sir_imp_part * total_part
                        sdr_imp_all += sdr_imp_part * total_part
                        sir_all_correct += sir_part_correct * correct_part
                        sdr_all_correct += sdr_part_correct * correct_part
                        sir_imp_all_correct += sir_imp_part_correct * correct_part
                        sdr_imp_all_correct += sdr_imp_part_correct * correct_part
                        correct_num += correct_part
                        total_num += total_part
                out[rkg_meth][sep_meth][tag]['SIR'] = [
                    [sir_all_correct / correct_num, sir_imp_all_correct / correct_num, correct_num],
                    [sir_all / total_num, sir_imp_all / total_num, total_num]
                ]
                out[rkg_meth][sep_meth][tag]['SDR'] = [
                    [sdr_all_correct / correct_num, sdr_imp_all_correct / correct_num, correct_num],
                    [sdr_all / total_num, sdr_imp_all / total_num, total_num]
                ]
    return out


def main(state=0, prefix=None):
    if state == 0:
        datafile_path = [
            prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study/sep_ret--ILRMA.pkl',
            # prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study/sep_ret--GMM.pkl',
            prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study/sep_ret--MVAE_onehot.pkl',
            prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study/sep_ret--MVAE_ge2e.pkl',
        ]
        metric = ["SDR", "SIR"][1]
        key = ['t60', 'angle interval'][0]
        out_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study/fig--sep_ret--{}--{}--only_angle_110.png'.format(metric, key)
        ret = extract_sepdata_fromfile(datafile_path, key=key, other="110")
        title = "{} results for different {}".format(metric, key)
        xlabel = 'T60 [s]' if key == 't60' else 'Angle interval [degree]'
        ylabel = "{} [dB]".format(metric)
        xticklabel = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65] if key == 't60' else [20, 30, 40, 70, 90, 110]
        xticklabel = list(map(str, xticklabel))
        ylim = None
        yticks = None
        legend = ['ILRMA', 'MVAE_1', "MVAE_2"]
        legend_pos = 'upper right'
        dis = 2.5
        plot_bar(ret[metric], out_path,
                 title=title, xlabel=xlabel, ylabel=ylabel, xticklabel=xticklabel,
                 yticks=yticks, ylim=ylim,
                 legend=legend, legend_pos=legend_pos, dis=dis)
    
    elif state == 1:
        """ ret: [[data for legend 0], [data for legend 1], ...] """
        file_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study/rkg_hist_ret--ILRMA--xvec_all_SDR.pkl'
        out_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study/fig-rkg_hist_ret--ILRMA--xvec_all_SDR.png'
        with open(file_path, 'rb') as f:
            out = pickle.load(f)
        title = None
        xlabel = "SDR [dB]"
        ylabel = "Accuracy [%]"
        xticklabel = ["[-8, -4]", "[-4, 0]", "[0, 4]", "[4, 8]", "[8, 12]", "[12, 16]", "[16, 20]", "[20, 24]"]
        # xticklabel = ["[-8, -4]", "[-4, 0]", "[0, 4]", "[4, 8]", "[8, 12]", "[12, 16]", "[16, 20]", "[20, 24]", "[24, 28]"]
        ylim = [0, 110]
        yticks = np.arange(0, 120, 20)
        legend = ["Orignal", "Original PLDA Aug", "Extractor Aug PLDA Aug"]
        # legend = ["Orignal", "Extractor Aug"]
        dis = 3
        legend_pos = "lower right"
        ret = []
        ret.append((out["xvec_withoutaug-plda_withoutaug_onlyln"] * 100).tolist())
        ret.append((out["xvec_withoutaug-plda_withaug_onlyln"] * 100).tolist())
        ret.append((out["xvec_withaug-plda_withaug_onlyln"] * 100).tolist())
        # ret.append((out["ge2e_withoutaug-cossim"] * 100).tolist())
        # ret.append((out["ge2e_withaug-cossim"] * 100).tolist())
        plot_bar(ret, out_path,
                 title=title, xlabel=xlabel, ylabel=ylabel, xticklabel=xticklabel,
                 yticks=yticks, ylim=ylim,
                 legend=legend, legend_pos=legend_pos, dis=dis)
    
    elif state == 2:
        file_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study_withsir/final_rets'
        out_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_angle_interval_study_withsir/fig/simu_extract_performance.png'
        rkg_method = [
            'xvec_withaug-plda_withaug_onlyln',
            # 'xvec_withoutaug-plda_withoutaug_onlyln',
        ]
        with open(os.path.join(file_path, 'rkg_ret.pkl'), 'rb') as f:
            out_rkg = pickle.load(f)
        with open(os.path.join(file_path, 'sep_ret.pkl'), 'rb') as f:
            out_sep = pickle.load(f)

        sep_method = ['ILRMA', 'MVAE_onehot', 'MVAE_onehot_ilrmainit']
        key = 'angle'
        if key == 't60':
            label = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
        elif key.startswith('angle'): 
            label = [20, 30, 40, 70, 90, 110]

        out_rkg_cat = concate_env_in_rkg(out_rkg, key=key, other=None)
        out_sep_cat = concate_env_in_sep(out_sep, key=key, other=None)
        data = {}
        data['bar_data'] = [[out_rkg_cat[rkg_method[0]][meth][env][2] * 100 for env in label] for meth in sep_method]
        data['line_data'] = [[out_sep_cat[rkg_method[0]][meth][env]['SIR'][0][0] for env in label] for meth in sep_method]

        xlabel = "DOA interval [degree]"
        ylabel_bar = "Accuracy [%]"
        ylabel_line = "SIR [dB]"
        xticklabel = label
        ylim_bar = [70, 140]
        yticks_bar = np.arange(80, 120, 20)
        ylim_line = [0, 22]
        yticks_line = np.arange(0, 20, 4)
        dis = 2.5
        legend = ['ILRMA', 'MVAE 1', 'MVAE 2']
        # legend_pos = "lower right"
        
        plot_bar_line(data, out_path, 
                      title=None, xlabel=xlabel, xticklabel=xticklabel,
                      ylabel_bar=ylabel_bar, ylim_bar=ylim_bar, yticks_bar=yticks_bar,
                      ylabel_line=ylabel_line, ylim_line=ylim_line, yticks_line=yticks_line,
                      legend=legend, legend_pos=None, dis=dis)

    elif state == 3:
        file_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_study/All-rkg_ret--ILRMA--ge2e.pkl'
        out_path = prefix + '/DATASET/SEPARATED_LIBRISPEECH/TEST_DATA/test_clean_simu/t60_study/fig-rkg_ret--ILRMA--ge2e.png'
        with open(file_path, 'rb') as f:
            out = pickle.load(f)
        legend = ['ILRMA', 'MVAE 1', 'MVAE 2']
        xlabel = "T60 [s]"
        ylabel = "Accuracy"
        xticklabel = ["0.160", "0.360", "0.610"]
        ylim = [0, 110]
        yticks = np.arange(0, 120, 20)
        dis = 2.5
        legend_pos = "lower right"
        plot_bar_line(data)
        
        


if __name__ == "__main__":
    # prefix = '/home/user/zhaoyi.gu/mnt/g2' 
    prefix = '/data/hdd0/zhaoyigu/' 
    main(state=2, prefix=prefix)

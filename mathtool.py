import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def H(X):
    """
    计算信息熵
    H(X) = -sigma p(x)log p(x)
    :param X:
    :return:
    """
    x_values = {}
    for x in X:
        x_values[x] = x_values.get(x, 0) + 1
    length = len(x_values)
    ans = 0
    for x in X:
        p = x_values.get(x) / length
        ans += p * math.log2(p)

    return 0 - ans


def condition_H(X, Y):
    """
    条件互信息计算
    H(X|Y) = Sigma_Y p(y)*H(X|Y=y)
    :param X:
    :param Y:
    :return:
    """
    y_value = set(Y)
    y_condition = {}
    for v in y_value:
        y_condition[v] = []
    for x, y in zip(X, Y):
        y_condition[y].append(x)
    y_counts = {}
    for y in Y:
        y_counts[y] = y_counts.get(y, 0) + 1
    ans = 0
    for k in y_counts:
        ans += k * H(y_condition[k])
    return ans / len(y_counts)


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算Gram核矩阵
    source: sample_size_1 * feature_size 的数据
    target: sample_size_2 * feature_size 的数据
    kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
    kernel_num: 表示的是多核的数量
    fix_sigma: 表示是否使用固定的标准差
    return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                    矩阵，表达形式:
                    [   K_ss K_st
                        K_ts K_tt ]
    """
    n_samples = source.shape[0] + target.shape[0]
    total = torch.cat([source, target], dim=0)  # 合并在一起

    total0 = total.unsqueeze(0).expand(total.shape[0],
                                       total.shape[0],
                                       total.shape[1])
    total1 = total.unsqueeze(1).expand(total.shape[0],
                                       total.shape[0],
                                       total.shape[1])
    L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|
    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val)  # 将多个核合并在一起


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n = source.shape[0]
    m = target.shape[0]

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss矩阵，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st矩阵，Source<->Target
    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts矩阵,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt矩阵,Target<->Target

    loss = (XX + XY).sum() + (YX + YY).sum()
    return loss


import prettytable as pt
import os
import datetime


class Statistics:
    """
    @@@@评估分类结果@@@@
    TstkSnhx
    """

    def __init__(self, y_hat, y, sensitives, threshold=0.5):

        self.hat_probability = y_hat
        self.true_label = y
        self.hat_label = np.where(y_hat > threshold, 1, 0)
        self.sensitives = [sensitives[:, i] for i in range(sensitives.shape[1])]
        self.order = "Tpr, Fpr, Tnr, Fnr"
        caps = []
        caps_ = []
        tp_ids = np.where(np.logical_and(self.true_label > 0, self.hat_label > 0))[0]
        fp_ids = np.where(np.logical_and(self.true_label < 1, self.hat_label > 0))[0]
        tn_ids = np.where(np.logical_and(self.true_label < 1, self.hat_label < 1))[0]
        fn_ids = np.where(np.logical_and(self.true_label > 0, self.hat_label < 1))[0]
        tp = tp_ids.shape[0]
        fp = fp_ids.shape[0]
        tn = tn_ids.shape[0]
        fn = fn_ids.shape[0]
        self.total = [tp, fp, tn, fn]
        value = [
            [], [], []
        ]
        value_ = [
            [], [], []
        ]
        ks = []
        for i in range(sensitives.shape[1]):
            s = sensitives[:, i]
            tps = np.count_nonzero(s[tp_ids])
            fps = np.count_nonzero(s[fp_ids])
            tns = np.count_nonzero(s[tn_ids])
            fns = np.count_nonzero(s[fn_ids])
            tps_ = np.count_nonzero(s[tp_ids] == 0)
            fps_ = np.count_nonzero(s[fp_ids] == 0)
            tns_ = np.count_nonzero(s[tn_ids] == 0)
            fns_ = np.count_nonzero(s[fn_ids] == 0)
            caps.append([tps, fps, tns, fns])
            caps_.append([tps_, fps_, tns_, fns_])

            s_1 = np.where(s == 1)
            s_0 = np.where(s == 0)

            ks.append([self.hat_probability[s_0], self.hat_probability[s_1]])
            sdp_ = np.mean(self.hat_probability[s_0])
            sdp = np.mean(self.hat_probability[s_1])
            pos_s_1 = np.where(np.logical_and(self.true_label == 1, s == 1))
            pos_s_0 = np.where(np.logical_and(self.true_label == 1, s == 0))

            pv_ = np.mean(self.hat_probability[pos_s_0])
            pv = np.mean(self.hat_probability[pos_s_1])

            neg_s_1 = np.where(np.logical_and(self.true_label == 0, s == 1))
            neg_s_0 = np.where(np.logical_and(self.true_label == 0, s == 0))

            nv_ = np.mean(self.hat_probability[neg_s_0])
            nv = np.mean(self.hat_probability[neg_s_1])
            value[0].append(sdp)
            value[1].append(pv)
            value[2].append(nv)
            value_[0].append(sdp_)
            value_[1].append(pv_)
            value_[2].append(nv_)

        self.caps = np.array(caps)
        self.caps_ = np.array(caps_)
        self.y_prob = ks

        tpr = self.caps[:, 0] / (self.caps[:, 0] + self.caps[:, 3])
        tpr_ = self.caps_[:, 0] / (self.caps_[:, 0] + self.caps_[:, 3])

        tnr = self.caps[:, 2] / (self.caps[:, 2] + self.caps[:, 1])
        tnr_ = self.caps_[:, 2] / (self.caps_[:, 2] + self.caps_[:, 1])

        num = self.caps.sum(1)
        num_ = self.caps_.sum(1)

        acc = (self.caps[:, 2] + self.caps[:, 0]) / num
        acc_ = (self.caps_[:, 2] + self.caps_[:, 0]) / num_

        "dp=pr"
        pr = (self.caps[:, 1] + self.caps[:, 0]) / num
        pr_ = (self.caps_[:, 1] + self.caps_[:, 0]) / num_

        'per'
        per = self.caps[:, 0] / (self.caps[:, 0] + self.caps[:, 1])
        per_ = self.caps_[:, 0] / (self.caps_[:, 0] + self.caps_[:, 1])

        'f1'
        # 2/(2+(fp+fn)/tp)
        "Tpr, Fpr, Tnr, Fnr"
        f1 = 2 / (2 + (self.caps[:, 1] + self.caps[:, 3]) / self.caps[:, 0])
        f1_ = 2 / (2 + (self.caps_[:, 1] + self.caps_[:, 3]) / self.caps_[:, 0])

        self.tpr = tpr
        self.tpr_ = tpr_
        self.tnr = tnr
        self.tnr_ = tnr_
        self.acc = acc
        self.acc_ = acc_
        self.dp = pr
        self.dp_ = pr_
        self.v = value
        self.v_ = value_
        self.f1 = f1
        self.f1_ = f1_
        self.dic = {}

        def merge(arr1, arr2):
            return [(i, j) for i, j in zip(arr1, arr2)]

        self.dic['Acc'] = merge(self.acc, self.acc_)
        self.dic['Tpr'] = merge(self.tpr, self.tpr_)
        self.dic['Tnr'] = merge(self.tnr, self.tnr_)
        self.dic['Dp'] = merge(self.dp, self.dp_)
        self.dic['f1'] = merge(self.f1, self.f1_)
        self.dic['Vdp'] = merge(self.v[0], self.v_[0])
        self.dic['Vp'] = merge(self.v[1], self.v_[1])
        self.dic['Vn'] = merge(self.v[2], self.v_[2])

        self.temp = {'ttacc': (self.total[0] + self.total[2]) / sum(self.total),
                     'f1': 2 / (2 + (self.total[1] + self.total[3]) / self.total[0])}

    def distribution(self, s=0):
        prob_0, prob_1 = self.y_prob[s]

        sns.set()
        plt.figure()
        sns.kdeplot(prob_0, shade=True, label='s{}=0'.format(s))
        sns.kdeplot(prob_1, shade=True, label='s{}=1'.format(s))
        plt.title('predict distributions for diff s{}'.format(s), fontsize='15')
        plt.xlim(0, 1)
        plt.ylim(0, 7)
        plt.yticks([])
        plt.ylabel('Prediction distribution')
        plt.show()

    def mmd(self, s=0):
        prob_0, prob_1 = self.y_prob[s]
        a = torch.from_numpy(prob_0).view(-1, 1)
        b = torch.from_numpy(prob_1).view(-1, 1)
        print(mmd(a, b))

    def detail(self):
        def merge(arr1, arr2):
            return ['{:<.4f}  {:0<.4f}'.format(i, j) for i, j in zip(arr1, arr2)]

        def calculate(arr, fz, fm):
            x = 0
            for i in fz:
                x += arr[i]
            y = 0
            for i in fm:
                y += arr[i]
            return '{:<.4f}'.format(x / y)

        res = ['s{}'.format(i) for i in range(len(self.sensitives))]
        pr = pt.PrettyTable(['Metrics', *res, 'all'])

        # tp fp tn fn
        pr.add_row(['Acc', *merge(self.acc, self.acc_), calculate(self.total, [0, 2], [0, 1, 2, 3])])
        pr.add_row(['Tpr', *merge(self.tpr, self.tpr_), calculate(self.total, [0], [0, 3])])
        pr.add_row(['Tnr', *merge(self.tnr, self.tnr_), calculate(self.total, [2], [1, 2])])
        pr.add_row(['Dp', *merge(self.dp, self.dp_), calculate(self.total, [0, 1], [0, 1, 2, 3])])
        pr.add_row([' '] * (2 + len(res)))
        pr.add_row(['Vdp', *merge(self.v[0], self.v_[0]), 'sum(h)'])
        pr.add_row(['Vp', *merge(self.v[1], self.v_[1]), 'sum(h|y=1)'])
        pr.add_row(['Vn', *merge(self.v[2], self.v_[2]), 'sum(h|y=0)'])
        pr.align = 'l'

        print(pr)

    def diff_fair(self):
        return self.info()

    def ratio_fair(self):
        return self.info(1)

    def info(self, m=0):

        tip = 'Diff' if m == 0 else 'Ratio'
        print('acc: {}. {} table:'.format(self.temp['ttacc'], tip))
        res = ['s{}'.format(i) for i in range(len(self.sensitives))]
        pr = pt.PrettyTable(['Metrics', *res])

        def sm(ls):
            def ks(i, j):
                h = i / j
                if h > 1:
                    h = 1 / h
                return h

            if m == 0:
                return ['{:.6f}'.format(abs(i - j)) for i, j in ls]
            else:
                return ['{:.6f}'.format(ks(i, j)) for i, j in ls]

        ck = ['Dp', 'Vdp', 'Tpr', 'Vp']

        for k in ck:
            pr.add_row([k, *sm(self.dic[k])])
        pr.align = 'l'

        print(pr)

    def metrics_help(self):
        um = {}
        um['Acc'] = 'accuracy'
        um['Tpr'] = 'equal opportunity'
        um['Tnr'] = 'equal opportunity'
        um['Dp'] = 'equal opportunity'
        um['f1'] = 'equal opportunity'
        um['Vdp'] = 'equal opportunity'
        um['Vp'] = 'equal opportunity'
        um['Vn'] = 'equal opportunity'


class Group:
    def __init__(self, prob_data, label_data, object_index, supple_index, sp=0.5):
        objects = []
        for index in object_index:
            objects.append([prob_data[index], label_data[index]])
        supple = []
        for index in supple_index:
            supple.append([prob_data[index], label_data[index]])

        objects = np.array(objects)
        supple = np.array(supple)

        self.intra_label = objects[:, 1]
        self.intra_prob = objects[:, 0]
        self.other_label = supple[:, 1]
        self.other_prob = supple[:, 0]
        self.intra_pred = np.where(self.intra_prob > sp, 1, 0)
        self.other_pred = np.where(self.other_prob > sp, 1, 0)

        sdp_ = np.mean(supple[:, 0])
        sdp = np.mean(objects[:, 0])
        self.sdp = [sdp, sdp_]

        pv = np.mean(objects[np.where(self.intra_label > 0)][:, 0])
        pv_ = np.mean(supple[np.where(self.other_label > 0)][:, 0])

        self.stp = [pv, pv_]

        pv = np.mean(objects[np.where(self.intra_label < 1)][:, 0])
        pv_ = np.mean(supple[np.where(self.other_label < 1)][:, 0])

        self.stn = [pv, pv_]

        acc = np.mean(self.intra_pred == self.intra_label)
        acc_ = np.mean(self.other_pred == self.other_label)
        self.acc = [acc, acc_]

        dp = np.mean(self.intra_pred == 1)
        dp_ = np.mean(self.other_pred == 1)
        self.dp = [dp, dp_]
        tpr = np.mean(self.intra_pred[np.where(self.intra_label > 0)] == 1)
        tpr_ = np.mean(self.other_pred[np.where(self.other_label > 0)] == 1)
        tnr = np.mean(self.intra_pred[np.where(self.intra_label < 1)] == 0)
        tnr_ = np.mean(self.other_pred[np.where(self.other_label < 1)] == 0)
        self.tpr = [tpr, tpr_]
        self.tnr = [tnr, tnr_]

    def __str__(self):
        pr = pt.PrettyTable(['Metrics', 'intra', 'other'])
        # tp fp tn fn
        pr.add_row(['Acc', *self.acc])
        pr.add_row(['Dp', *self.dp])
        pr.add_row(['Tpr', *self.tpr])
        pr.add_row(['Tnr', *self.tnr])

        pr.add_row([' '] * 3)
        pr.add_row(['Vdp', *self.sdp])
        pr.add_row(['Vp', *self.stp])
        pr.add_row(['Vn', *self.stn])
        pr.align = 'l'
        return pr.__str__()

    def numpy(self):
        """
        ACC, DP, TRP, TNR, VDP, VP, VN
        :return:
        """
        ls = [[*self.acc],
              [*self.dp],
              [*self.tpr],
              [*self.tnr],
              [*self.sdp],
              [*self.stp],
              [*self.stn],
              ]
        return np.array(ls)

    def diff(self):
        ks = self.numpy()
        return np.abs(ks[:, [1]] - ks[:, [0]])

    def ratio(self):
        ks = self.numpy()
        m = ks[:, [1]] / ks[:, [0]]
        m = np.where(m > 1, 1 / m, m)
        return m


import pickle


class Statics(object):
    def __init__(self, y_hat, y, sensitives, threshold=0.5):

        if torch.is_tensor(y_hat):
            y_hat = y_hat.cpu().numpy()
        if torch.is_tensor(sensitives):
            sensitives = sensitives.cpu().numpy()
        if torch.is_tensor(y):
            y = y.cpu().numpy()

        self.parts = sensitives.mean(0)

        self.result = {}
        for i in range(sensitives.shape[1]):
            ids = np.where(sensitives[:, i] == 1)[0]
            ids_ = np.where(sensitives[:, i] == 0)[0]

            g = Group(y_hat, y, ids, ids_, threshold)
            self.result[i] = g
        self.label = y
        self.prob = y_hat
        self.pred = np.where(y_hat > threshold, 1, 0)
        self.sdp = np.mean(y_hat)
        self.stp = np.mean(y_hat[np.where(y > 0)])
        self.stn = np.mean(y_hat[np.where(y < 1)])
        self.acc = np.mean(y == self.pred)
        self.dp = np.mean(self.pred == 1)
        self.tpr = np.mean(self.pred[np.where(self.label > 0)] == 1)
        self.tnr = np.mean(self.pred[np.where(self.label < 1)] == 0)

    def diff(self, item=None):
        if item is None:
            item = ['acc', 'dp', 'tpr', 'tnr', 'vdp']
        print(self.__info(item))

    def ratio(self, item=None):
        if item is None:
            item = ['acc', 'dp', 'tpr', 'tnr', 'vdp']
        print(self.__info(item, mod=0))

    def info(self, item, m):
        print(self.__info(item, m))

    def __info(self, item, mod=1):
        items = ['acc', 'dp', 'tpr', 'tnr', 'vdp', 'vp', 'vn']
        ls = {}
        for k in self.result:
            g = self.result[k]
            if mod:
                c = g.diff()
            else:
                c = g.ratio()
            for i, it in enumerate(items):
                ls[it] = ls.get(it, [])
                ls[it].append(round(c[i][0], 8))
        m = 'diff' if mod else 'ratio'
        res = ['s{}'.format(i) for i in range(len(self.result))]
        pr = pt.PrettyTable(['Metrics', *res])
        for it in item:
            pr.add_row([f'{m} {it}', *ls[it]])
        pr.align = 'l'
        return pr

    def __str__(self):
        pr = pt.PrettyTable(['Metrics', 'value'])
        # tp fp tn fn
        pr.add_row(['Acc', self.acc])
        pr.add_row(['Dp', self.dp])
        pr.add_row(['Tpr', self.tpr])
        pr.add_row(['Tnr', self.tnr])

        pr.add_row([' '] * 2)
        pr.add_row(['Vdp', self.sdp])
        pr.add_row(['Vp', self.stp])
        pr.add_row(['Vn', self.stn])
        pr.align = 'l'
        return pr.__str__()

    def detail(self):
        print(*self.__detail(), sep='\n')

    def __detail(self):
        ls = [self]
        for k in self.result:
            g = self.result[k]
            ls.append(g.__str__())
        return ls

    def information(self, s=0):
        print(self.acc)
        self.distribution(s)
        self.diff()
        self.mmd(s)

    @staticmethod
    def getpath(**exp_args):
        keys = {"dataset": 'adult',
                "method": 'fvc',
                "seed": 1,
                "sensitive": 0,
                'parameters': []}
        args = exp_args
        for k in keys:
            args[k] = args.get(k, keys[k])
        dir = 'exp/{}_{}_s{}'.format(args['dataset'], args['method'], args['sensitive'])
        if not os.path.exists(dir):
            os.mkdir(dir)

        t = datetime.date.today()
        path = 'seed{}_{:0>2}{:0>2}'.format(args['seed'], t.month, t.day)
        path_ = f'{dir}/{path}'
        i = 1
        while True:
            if os.path.exists(path_):
                path_ = f'{dir}/{path}' + f'_{i}'
            else:
                break
            i += 1
        return path_

    def save_result(self, **exp_args):

        """
        保存实验结果
        :param message:
        :return:
        """
        path_ = Statics.getpath(**exp_args)
        ra = self.__info(['acc', 'dp', 'tpr', 'tnr', 'vdp', 'vp', 'vn'], 0).__str__()
        di = self.__info(['acc', 'dp', 'tpr', 'tnr', 'vdp', 'vp', 'vn'], 1).__str__()
        ps = ''
        for it in self.__detail():
            ps += str(it) + '\n'
        res_path = f'{path_}.pkl'
        ls = [str(exp_args.get('parameters', [])), f'acc:{self.acc}', ra, di, 'detail', ps, f'result @ {res_path}']
        with open(path_, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(ls))

        with open(res_path, 'wb') as f:  # 打开文件
            pickle.dump(self, f)
        print(f'success saved @ {path_}')

    @staticmethod
    def load_result(path):
        with open(path, 'rb') as f:  # 打开文件
            re = pickle.load(f)
        return re

    def distribution(self, s=0):
        prob_0, prob_1 = self.result[s].intra_prob, self.result[s].other_prob
        plt.figure()
        sns.kdeplot(prob_0, shade=True, label='s{}=0'.format(s))
        sns.kdeplot(prob_1, shade=True, label='s{}=1'.format(s))
        plt.title('predict distributions for diff s{}'.format(s), fontsize='15')
        plt.xlim(0, 1)
        plt.ylim(0, 7)
        plt.yticks([])
        plt.ylabel('Prediction distribution')
        plt.show()

    def mmd(self, s=0):
        prob_0, prob_1 = self.result[s].intra_prob, self.result[s].other_prob
        a = torch.from_numpy(prob_0).view(-1, 1)
        b = torch.from_numpy(prob_1).view(-1, 1)
        print('mmd:', mmd(a, b))

    def clear_out(self, s):
        return self.acc, self.result[s].diff()[1][0], self.result[s].diff()[4][0], \
               self.result[s].diff()[2][0], self.result[s].diff()[3][0]


class Experiment:
    def __init__(self, note=None, **exp_args):
        """

        :param note:
        :param exp_args:
        includes:
             dataset:
             method:
             seed:
             sensitive:
             parameters:
        """
        print(exp_args)
        dataset = exp_args['dataset']
        method = exp_args['method']

        if exp_args.get('path'):
            self.path = exp_args.get('path')
        else:
            if not os.path.exists(f'exp/{dataset}'):
                os.mkdir(f'exp/{dataset}')
            self.path = f'exp/{dataset}/{method}'
        self.note = note
        keys = {"dataset": 'adult',
                "method": 'fvc',
                "seed": 1,
                "sensitive": 0,
                'parameters': [],
                'abbr': ''}
        self.args = exp_args
        for k in keys:
            self.args[k] = self.args.get(k, keys[k])
        parameters = {}
        for i, k in enumerate(self.args['parameters']):
            parameters[f'p{i}'] = k
        if self.args['method'] == 'FNNC——':
            parameters[f'p{1}'] = self.args['parameters'] if type(self.args['parameters']) is str else 'ce'


        self.parameters = parameters
        if self.args['sensitive'] == 'auto':
            self.args['sensitive'] = 0

    def append(self, result: Statics):
        """
        dataset, method, seed, acc, dp, sdp, eo_p, eo_n, *method_args
        diff()@ ACC, DP, TRP, TNR, VDP, VP, VN
        :param result:
        :return:
        """

        tu = []
        for k in result.result:
            tu.append(result.result[k].diff())

        tu = np.hstack(tu)

        ACC, DP, TPR, TNR, VDP, VP, VN = tu
        diff = [DP, VDP, TPR, TNR, VP, VN]
        info = [self.args['dataset'],
                self.args['method'],
                self.args['seed'], result.acc,
                *list(map(lambda x: (x * result.parts).sum(), diff))]
        for k in self.parameters:
            info.append(self.parameters[k])

        info = ' '.join(list(map(str, info)))
        with open(self.path, 'a+', encoding='utf-8') as f:
            f.write(info)
            f.write('\n')
        print(f'append data: \n{info}')



    def exp(self):
        """
        read 动态 exp
        :return:
        """
        dt = pd.read_csv(self.path, sep=' ', names=['dataset', 'method', 'seed', 'acc',
                                                    'dp', 'sdp', 'eop', 'eon', 'seop', 'seon', *self.parameters.keys()])
        return dt

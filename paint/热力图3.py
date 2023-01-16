from methods.FRC import FRC
import seaborn as sns
from torch.nn import Linear
import numpy as np
import torch
import tools
from sklearn.linear_model import LogisticRegression
import mathtool
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mlp




# 4589
def fun(x, y, a):
    a1 = a[:, 4]
    a2 = a[:, 5]
    a3 = a[:, 8]
    a4 = a[:, 9]
    idx = torch.where(torch.logical_or(a1 == 1, a2 == 1))
    size = idx[0].shape[0] // 2
    idx3 = torch.where(a3 == 1)[0][:size]
    idx4 = torch.where(a4 == 1)[0][:size]

    ans_x = torch.vstack([x[idx], x[idx3], x[idx4]])
    ans_y = torch.hstack([y[idx], y[idx3], y[idx4]])
    ans_a = torch.vstack([a[idx], a[idx3], a[idx4]])
    ans_a = torch.hstack([ans_a[:, [4]],
                          ans_a[:, [5]],
                          ans_a[:, [8]],
                          ans_a[:, [9]]])
    return ans_x, ans_y, ans_a


def exp(seed, mm='A'):
    tools.seed_everything(seed)
    m = f'FRC-{mm}'
    if m == 'FRC-A':
        model = FRC(inputs_size=data_d['test'][0].shape[1],
                    parameters=[0.01, 0.01],
                    z_dim=20,
                    device='cpu',
                    sens=list(range(4))
                    )
    elif m == 'FRC-M':
        model = FRC(inputs_size=data_d['test'][0].shape[1],
                    parameters=[0.01, 0.01], z_dim=20,
                    CORR=[3], irr=[3] * 19,
                    device='cpu',
                    )
    else:
        raise NotImplementedError
    model.fit(data_d, 100)
    ms = model.test(data_d)
    exp = mathtool.Experiment(dataset='adult',
                              method=m,
                              seed=seed,
                              sensitive='auto',
                              parameters=[0.01, 0.01],
                              abbr='')
    exp.append(ms)


if __name__ == '__main__':
    ps = '../npz_data/adult_s_10.npz'
    data_d = tools.load_single(ps)

    x, y, a = fun(*data_d['train'])
    xx, yy, aa = fun(*data_d['test'])
    print(x.shape, a.shape, a.sum(dim=0))
    print(xx.shape, aa.shape, aa.sum(dim=0))
    print(x.shape[0]+xx.shape[0])
    tensor_dataset = torch.utils.data.TensorDataset(x, y, a)
    train_load = torch.utils.data.DataLoader(tensor_dataset, batch_size=128, shuffle=True)
    data_d = {
        'train': (x, y, a),
        'test': (xx, yy, aa),
        'loader': train_load,

    }
#     for s in range(10):
#         print('FRC-A:', s)
#         exp(s, 'A')
#     for s in range(10):
#         print('FRC-M:', s)
#         exp(s, 'M')
"""

+----------+------------+------------+------------+------------+
| Metrics  | s0         | s1         | s2         | s3         |
+----------+------------+------------+------------+------------+
| diff acc | 0.11378171 | 0.01129614 | 0.07523278 | 0.17270928 |
| diff dp  | 0.00189753 | 0.00197628 | 0.00193424 | 0.00581395 |
| diff tpr | 0.01       | 0.01234568 | 0.0106383  | 0.01960784 |
| diff tnr | 0.0        | 0.0        | 0.0        | 0.0        |
| diff vdp | 0.05838904 | 0.03353747 | 0.04558397 | 0.0667189  |
+----------+------------+------------+------------+------------+
acc: 0.8388969521044993
{'dataset': 'adult', 'method': 'FNNC', 'seed': 0, 'sensitive': 'auto', 'parameters': [0.01, 1], 'abbr': ''}
append data: 
adult FNNC 0 0.8388969521044993 0.0029052961543647895 0.050671228423226904 0.013180835297707288 0.0 0.08431577414470386 0.03928413629600918 0.01 1

"""

"""
frc-a
adult FNNC 1 0.8403483309143687 0.0037530178062409932 0.06348024060660426 0.01610510578895802 0.0 0.10572485151235622 0.049347693815916596 0.01 1
+----------+------------+------------+------------+------------+
| Metrics  | s0         | s1         | s2         | s3         |
+----------+------------+------------+------------+------------+
| diff acc | 0.11188418 | 0.00583166 | 0.07329855 | 0.17464352 |
| diff dp  | 0.00379507 | 0.0034882  | 0.00386847 | 0.00387972 |
| diff tpr | 0.02       | 0.01991239 | 0.0212766  | 0.0032144  |
| diff tnr | 0.0        | 0.0        | 0.0        | 0.0        |
| diff vdp | 0.07083638 | 0.03693545 | 0.05936373 | 0.08891072 |
+----------+------------+------------+------------+------------+

"""

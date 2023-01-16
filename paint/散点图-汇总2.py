import mathtool
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

col = ['dataset', 'method',
       'seed', 'acc', 'dp',
       'sdp', 'eop', 'eon',
       'seop', 'seon', 'p0',
       'p1']


def get_pandas(files):
    dt = []
    for f in files:
        f = f'exp/{f.split("_")[0]}/{f.split("_")[1]}'
        dt.append(pd.read_csv(f, sep=' ', names=col))
    return dt


def get_data(x, y):
    ans = {}
    for i, j in zip(x, y):
        key = fx(i)
        ans[key] = ans.get(key, [])
        ans[key].append(j)
    ress = []
    for k in ans:
        ls = ans[k]
        ress.append([k, *np.percentile(ls, (25, 50, 75))])

    return np.array(ress)


cash = np.linspace(0, 0.2, 41)
print(cash)


def fx(x):
    for i in cash:
        if x <= i:
            return i
    return round(x, 3)


def draw(d1, d2, m, color='pink', label=None):
    print(label, f'{d1.shape[0]} points')
    datas = get_data(d1.values, d2.values)
    print(datas)
    datas = datas[np.argsort(datas[:, 0])]
    print(color)
    plt.scatter(datas[:, 0], datas[:, 1], color=color,  marker=m, alpha=0.2)
    plt.scatter(datas[:, 0], datas[:, 2], color=color, marker=m, s=80, label=label, alpha=0.8)
    plt.scatter(datas[:, 0], datas[:, 3], color=color,  marker=m, alpha=0.2)
    # plt.fill_between(datas[:, 0], datas[:, 1], datas[:, 3], color=color, alpha=0.12)


def data_clean(dt, k, t):
    dt = dt[dt[k] <= t]
    return dt
def plant(file_compas):
    dlist = get_pandas(file_compas)
    colors = ['red','green', 'blue',  'purple', 'coral', 'r']
    m = ['D', '<', 's','o', 'H', '_']
    for i in range(len(file_compas)):
        path = file_compas[i]
        method = path.split('_')[-1]
        dt = dlist[i]
        print(dt)

        draw(dt[key], dt.acc,m[i], colors[i], method)

# draw(dt_adv.dp, dt_adv.acc, 'red', 'adv')
x = ['dp', 'eop', 'eon', 'sdp']
key = x[0]

file_h = ['heritage_FRC', 'heritage_FBC', 'heritage_Cfair','heritage_Fair-scale']
file_adult_M = ['adult-M_FRC-M','adult-M_FRC-A','adult-M_CFair','adult-M_FBC','adult-M_Fair-scale']
file_compas_M = ['compas-M_FRC-M', 'compas-M_FRC-A', 'compas-M_CFair', 'compas-M_FBC', 'compas-M_Fair-scale']


fontsize = 15

plt.figure(figsize=(9, 10))
plt.subplot(2, 1, 1)
plt.title(r'M.Compas', fontsize=fontsize)
plt.grid()
plant(file_compas_M)
plt.xlim(0, 0.12)
plt.plot()
plt.xlabel(r'$\Delta~DP$', fontsize=fontsize)
plt.ylabel(r'Accuracy', fontsize=fontsize)
plt.legend(loc=4,prop={'size': 16})

plt.subplot(2, 1, 2)
plt.title(r'M.Adult', fontsize=fontsize)
plt.grid()
plant(file_adult_M)

plt.ylim(None, 0.835)
plt.xlim(0, 0.12)
plt.plot()
plt.xlabel(r'$\Delta~DP$', fontsize=fontsize)
plt.ylabel(r'Accuracy', fontsize=fontsize)
plt.legend(loc=4, ncol=3, prop={'size': 16})


# plt.savefig('image_name',bbox_inches='tight')
plt.show()

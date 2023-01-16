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


cash = np.linspace(0, 0.4, 81)
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
import matplotlib.ticker as ticker
# draw(dt_adv.dp, dt_adv.acc, 'red', 'adv')
x = ['dp', 'eop', 'eon', 'sdp']
key = x[0]

file_h = ['heritage_FRC', 'heritage_FBC', 'heritage_Cfair','heritage_Fair-scale']


fontsize = 15

plt.figure(figsize=(9, 5))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
# plt.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
plt.title(r'Heritage', fontsize=fontsize)
plt.grid()
plant(file_h)

plt.xticks([0, 0.05, 0.1, 0.2, 0.3, 0.4])
plt.ylim(0.62, 0.84)
plt.xlim(0, 0.45)
plt.plot()
plt.xlabel(r'$\Delta~DP$', fontsize=fontsize)
plt.ylabel(r'Accuracy', fontsize=fontsize)
plt.legend(loc=4,prop={'size': 16})
plt.show()


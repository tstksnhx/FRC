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


def draw(d1, d2, color='pink', label=None):
    print(label, f'{d1.shape[0]} points')
    datas = get_data(d1.values, d2.values)
    print(datas)
    datas.sort(axis=0)
    sns.lineplot(datas[:, 0], datas[:, 1], color=color, alpha=0.1)
    sns.lineplot(datas[:, 0], datas[:, 2], color=color, label=label)
    sns.lineplot(datas[:, 0], datas[:, 3], color=color, alpha=0.1)
    plt.fill_between(datas[:, 0], datas[:, 1], datas[:, 3], color=color, alpha=0.12)


def data_clean(dt, k, t):
    dt = dt[dt[k] <= t]
    return dt


# draw(dt_adv.dp, dt_adv.acc, 'red', 'adv')
x = ['dp', 'eop', 'eon', 'sdp']
key = x[0]

file_compas = ['compas_frc', 'compas_corr', 'compas_fnnc']
file_adult = ['adult_frc', 'adult_corr', 'adult_fnnc', 'adult_fair-scale']
# compas_m = ['compas_m_frc', 'compas_m_fbc']
# adult_m = ['adult_m_frc', 'adult_m_fbc']

def plant(file_compas):
    dlist = get_pandas(file_compas)
    colors = ['red','green', 'blue',  'purple']
    for i in range(len(file_compas)):
        path = file_compas[i]
        method = path.split('_')[-1]
        dt = dlist[i]
        print(dt)

        draw(dt[key], dt.acc, colors[i], method)


fontsize = 15

plt.figure(figsize=(11, 5))

plt.subplot(1, 2, 1)
plt.title(r'Adult', fontsize=fontsize)

plant(file_adult)

plt.xlim(0, 0.06)
plt.ylim(None, 0.83)
plt.plot()
plt.xlabel(r'$\Delta~DP$', fontsize=fontsize)
plt.ylabel(r'Accuracy', fontsize=fontsize)
plt.legend(loc=2)

plt.subplot(1, 2, 2)
plt.title(r'Compas', fontsize=fontsize)
plant(file_compas)
plt.xlabel(r'$\Delta~DP$', fontsize=fontsize)
plt.ylabel(r'Accuracy', fontsize=fontsize)
# plt.xticks([0, 0.02, 0.04, 0.06])
plt.xlim(0, 0.06)
plt.plot()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


data = 'heritage'
split = 2
c12 = pd.read_csv(f'{data}_frc-a.csv', header=0, index_col=0)
c22 = pd.read_csv(f'{data}_vae.csv', header=0, index_col=0)
"""
ax1.set_title(r'FRC ($\Delta DP=0.0492)')
ax1.set_yticklabels(
    ax1.get_yticklabels(), rotation=0)

sns.heatmap(c22, ax=ax2, **heatmap_config)
ax2.set_title(r'$\beta$-VAE ($\Delta DP=0.1095)')

"""

def isShow(i):
    if i + 1 == split or i == 0:
        return f'z{i + 1}'
    if i + 1 == len(c12.columns):
        return f'z{i + 1}'
    if i % 4 == 3:
        return f'z{i + 1}'
    return ''


c12.columns = [f'z{i + 1}' if isShow(i) else ' ' for i in range(len(c12.columns))]
c22.columns = [f'z{i + 1}' if isShow(i) else ' ' for i in range(len(c12.columns))]

ls = [c12, c22]
vmin = min(c12.values.min(), c22.values.min())
vmax = max(c12.values.max(), c22.values.max())

heatmap_config = {
    'annot': False,
    # 'vmax': vmax,
    # 'vmin': vmin,
    'linewidths': 0.0,
    'cmap': 'RdPu',
    'fmt': '.3f',
    'cbar': False
}

fig, (ax1, ax2, ax3) = plt.subplots(
    figsize=(10, 3), ncols=1, nrows=3,
    gridspec_kw=dict(height_ratios=[1.5, 1.5, 1.2]))
# fig, (ax1, ax2, ax3) = plt.subplots(
#     figsize=(8, 1), ncols=1, nrows=3,
#     gridspec_kw=dict(height_ratios=[4, 4, 0.6]))
sns.heatmap(c12, ax=ax1, **heatmap_config)

ax1.set_title(r'FRC-A')
ax1.set_yticklabels(
    ax1.get_yticklabels(), rotation=0)

sns.heatmap(c22, ax=ax2, **heatmap_config)
ax2.set_title(r'$\beta$-VAE')
ax2.set_yticklabels(
    ax2.get_yticklabels(), rotation=0)

fig.colorbar(ax2.collections[0], cax=ax3, orientation='horizontal')
ax1.axvline(split, linewidth=4, linestyle="--", color='red')
ax2.axvline(split, linewidth=4, linestyle="--", color='red')
plt.show()

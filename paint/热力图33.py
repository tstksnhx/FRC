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





if __name__ == '__main__':
    ps = '../npz_data/adult_s_10.npz'
    data_d = tools.load_single(ps)

    x, y, a = data_d['train']
    xx, yy, aa = data_d['test']
    print(x.shape, a.shape, a.sum(dim=0))
    print(xx.shape, aa.shape, aa.sum(dim=0))
    print(x.shape[0]+xx.shape[0])

print((6021 + 13409 + 902 + 1865)/(22792+3256))
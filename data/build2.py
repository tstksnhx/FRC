import numpy as np

import data.data_build as bd
train = bd.heritage_create('heritage-train.csv', 120000)
val = bd.heritage_create('heritage-validate.csv')
test = bd.heritage_create('heritage-test.csv')

np.savez('heritage_train.npz', x=train['x'],
         y=train['y'], s=train['train_sensitive_sex'])

"""
val~ test~:
train_x
train_y
train_sensitive_sex
train_sensitive_age
train_sensitive_multi
"""

import numpy as np

import data.data_build as bd
train = bd.heritage_create('heritage-train.csv', 120000)
val = bd.heritage_create('heritage-validate.csv')
test = bd.heritage_create('heritage-test.csv')
print(train['x'].shape)
print(val['x'].shape)
print(test['x'].shape)
np.savez('heritage_train.npz', x=train['x'],
         y=train['y'], s=train['sensitive_age'])
np.savez('heritage_test.npz', x=test['x'],
         y=test['y'], s=test['sensitive_age'])

np.savez('heritage_verify.npz', x=val['x'],
         y=val['y'], s=val['sensitive_age'])

"""
val~ test~:
train_x
train_y
train_sensitive_sex
train_sensitive_age
train_sensitive_multi
"""

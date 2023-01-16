import numpy as np

import data.data_build as bd
train = bd.heritage_create('heritage-train.csv', 120000)
val = bd.heritage_create('heritage-validate.csv')
test = bd.heritage_create('heritage-test.csv')
npz_map = {}
for k in train:
    print(k)
    npz_map['train_'+k] = train[k]
for k in val:
    npz_map['val_'+k] = val[k]
for k in test:
    npz_map['test_'+k] = test[k]
print(npz_map['train_x'].shape)
# np.savez('heritae.npz', **npz_map)
# ld = np.load('heritage.npz')
# for k in ld:
#     print(k)

"""
val~ test~:
train_x
train_y
train_sensitive_sex
train_sensitive_age
train_sensitive_multi
"""

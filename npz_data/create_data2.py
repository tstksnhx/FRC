import dataset
import numpy as np


def make(name, t):
    dt = {'train': '../data/to-ml-compas/compas_train.csv',
          'validate': '../data/to-ml-compas/compas_test.csv',
          'test': '../data/to-ml-compas/compas_test.csv',
          'name': 'Compas',
          'seed': 1,
          'tsne': t}
    dt2 = {'train': '../data/to-ml-adult/adult.data',
           'validate': '../data/to-ml-adult/adult.validate',
           'test': '../data/to-ml-adult/adult.test_true',
           'name': 'AdultDataSet',
           'seed': 1,
           'tsne': t}
    if name == 'adult':
        dset_train = dataset.AdultDataSet.from_dict(dt2, type='train')
        dset_test = dataset.AdultDataSet.from_dict(dt2, type='test')
        dset_validation = dataset.AdultDataSet.from_dict(dt2, type='validate')
    elif name == 'compas':
        dset_train = dataset.Compas.from_dict(dt, type='train')
        dset_test = dataset.Compas.from_dict(dt, type='test')
        dset_validation = dataset.Compas.from_dict(dt, type='validate')
    else:
        raise NotImplemented()
    return dset_train, dset_test, dset_validation

dset_train, dset_test, dset_validation = make('compas', True)

np.savez('compas_s_1.npz',
         x_train=dset_train.data.values,x_test=dset_test.data.values,x_valid=dset_validation.data.values,
         y_train=dset_train.outcome.values, y_test=dset_test.outcome.values,y_valid=dset_validation.outcome.values,
         s_train=dset_train.sensitive.values, s_test=dset_test.sensitive.values,s_valid=dset_validation.sensitive.values,
         input_columns=list(dset_train.data.columns),
         s_columns=list(dset_train.sensitive.columns))



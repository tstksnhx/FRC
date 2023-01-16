import pandas as pd
import numpy as np


def heritage_create(path, size=0):
    data = pd.read_csv(path)
    print(data.shape, size)
    if size:
        # data = data.sample(frac=1., random_state=1)
        data = data.loc[:size, :]
        data.set_index(np.arange(len(data)), inplace=True)
    print(data.shape, size)
    data['CI'] = (data['CI'] != '0').astype('int32')
    data['PayDelay'] = data['PayDelay'].astype('category').cat.codes
    data = data[['ClaimsCount', 'PayDelay', 'LabCount', 'DrugCount', 'CI', 'AMI',
                 'APPCHOL', 'ARTHSPIN', 'CANCRA', 'CANCRB', 'CANCRM', 'CATAST', 'CHF',
                 'COPD', 'FLaELEC', 'FXDISLC', 'GIBLEED', 'GIOBSENT', 'GYNEC1', 'GYNECA',
                 'HEART2', 'HEART4', 'HEMTOL', 'HIPFX', 'INFEC4', 'LIVERDZ', 'METAB1',
                 'METAB3', 'MISCHRT', 'MISCL1', 'MISCL5', 'MSC2a3', 'NEUMENT', 'ODaBNCA',
                 'PERINTL', 'PERVALV', 'PNCRDZ', 'PNEUM', 'PRGNCY', 'RENAL1', 'RENAL2',
                 'RENAL3', 'RESPR4', 'ROAMI', 'SEIZURE', 'SKNAUT', 'STROKE',
                 'TRAUMA', 'UTI', 'ANES', 'EM', 'MED', 'PL', 'RAD', 'SAS', 'SCS', 'SDS',
                 'SEOA', 'SGS', 'SIS', 'SMCD', 'SMS', 'SNS', 'SO', 'SRS', 'SUS',
                 'AgeAtFirstClaim', 'Sex',
                 ]]

    for var in list(data.columns):
        data = data[~data[var].isnull()]
    print(data.shape)
    data.set_index(np.arange(len(data)), inplace=True)
    data['delay'] = (data.PayDelay > 0.5).astype('int32')
    sensitive_sex = pd.get_dummies(data['Sex'])
    age_inf = (data['AgeAtFirstClaim']
               .isin(['60-69', '70-79', '80+'])
               ).apply(lambda x: 'elder' if x == 1 else 'other')
    label = data['CI']
    sensitive_age = pd.get_dummies(age_inf)
    sensitive_multi = pd.get_dummies(data['Sex'] + '_' + age_inf)
    data.drop(['AgeAtFirstClaim', 'Sex', 'CI', 'delay'], axis=1, inplace=True)
    data = (data - data.mean()) / data.std()
    return {
        'x': data,
        'y': label,
        'sensitive_sex': sensitive_sex,
        'sensitive_age': sensitive_age,
        'sensitive_multi': sensitive_multi
    }


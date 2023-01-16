import numpy as np
import pandas as pd

"""
********************************
*********** data load **********
********************************
"""


def load_compas(path='dataset/compas/compas-scores-two-years.csv'):
    if path is None:
        path = 'dataset/compas/compas-scores-two-years.csv'
    """
    data source https://github.com/propublica/compas-analysis
    :param path:
    :return:
    """
    print(path)
    df = pd.read_csv(path)
    FEATURES = [
        'age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex',
        'priors_count', 'days_b_screening_arrest', 'decile_score', 'is_recid',
        'two_year_recid'
    ]
    df = df[FEATURES]
    df = df[df.days_b_screening_arrest <= 30]
    df = df[df.days_b_screening_arrest >= -30]
    df = df[df.is_recid != -1]
    df = df[df.c_charge_degree != 'O']
    df = df[df.score_text != 'N/A']
    continuous_features = [
        'priors_count', 'days_b_screening_arrest', 'is_recid', 'two_year_recid'
    ]
    continuous_to_categorical_features = ['age', 'decile_score', 'priors_count']
    categorical_features = ['c_charge_degree', 'race', 'score_text', 'sex']

    for c in continuous_to_categorical_features:
        b = [0] + list(np.percentile(df[c], [20, 40, 60, 80, 90, 100]))
        if c == 'priors_count':
            b = list(np.percentile(df[c], [0, 50, 70, 80, 90, 100]))
        df[c] = pd.cut(df[c], b, labels=False)
    df = pd.get_dummies(df, columns=categorical_features + continuous_to_categorical_features)
    to_fill = [
        u'decile_score_0', u'decile_score_1', u'decile_score_2',
        u'decile_score_3', u'decile_score_4', u'decile_score_5'
    ]
    for i in range(len(to_fill) - 1):
        df[to_fill[i]] = df[to_fill[i:]].max(axis=1)
    to_fill = [
        u'priors_count_0.0', u'priors_count_1.0', u'priors_count_2.0',
        u'priors_count_3.0', u'priors_count_4.0'
    ]
    for i in range(len(to_fill) - 1):
        df[to_fill[i]] = df[to_fill[i:]].max(axis=1)
    features = [u'c_charge_degree_F', u'c_charge_degree_M',
        u'race_African-American', u'race_Asian', u'race_Caucasian',
        u'race_Hispanic', u'race_Native American', u'race_Other',
        u'score_text_High', u'score_text_Low', u'score_text_Medium',
        u'sex_Female', u'sex_Male', u'age_0', u'age_1', u'age_2', u'age_3',
        u'age_4', u'age_5', u'decile_score_0', u'decile_score_1', u'decile_score_2',
        u'decile_score_3', u'decile_score_4', u'decile_score_5', u'priors_count_0.0', u'priors_count_1.0',
        u'priors_count_2.0', u'priors_count_3.0', u'priors_count_4.0'
    ]
    label = ['two_year_recid']
    df = df[features + label]
    LABEL_COLUMN = 'two_year_recid'
    PROTECTED_GROUPS = [
        'sex_Female', 'sex_Male', 'race_Caucasian', 'race_African-American'
    ]
    X = df[features]
    y = df[LABEL_COLUMN]
    S = df[PROTECTED_GROUPS]
    return X, y, S


def load_bank_(path='dataset/bank/bank.csv'):
    if path is None:
        path = 'dataset/bank/bank.csv'
    CATEGORICAL_COLUMNS = ['job', 'marital', 'month', 'contact', 'poutcome', 'education']
    fun = lambda x: 1 if x == 'yes' else 0
    SHIFT = [
        ['deposit', fun],
        ['loan', fun],
        ['housing', fun],
        ['default', fun],
        ['pdays', lambda x: 999 if x == -1 else x]
    ]
    CONTINUOUS_COLUMNS = [
        'age', 'balance', 'duration', 'pdays', 'day'
    ]
    CONTINUOUS_RULE = [5, 4, 4, [0, 30, 90, 180, 365, 450, 700, 1000], 5]

    re = easy_load(path,
                   cat=CATEGORICAL_COLUMNS,
                   con=CONTINUOUS_COLUMNS,
                   csp=CONTINUOUS_RULE, shift=SHIFT)
    features = list(re.keys())
    s = ['age_0', 'age_1', 'age_2', 'age_3', 'age_4']

    for item in s + ['deposit']:
        features.remove(item)

    S = re[s]
    y = re['deposit']
    X = re[features]
    return X, y, S


def load_adult(path='dataset/adult/adult.data'):
    if path is None:
        path = 'dataset/adult/adult.data'
    CATEGORICAL_COLUMNS = [
        'workclass', 'education', 'marital_status', 'occupation', 'relationship',
        'race', 'gender', 'native_country'
    ]
    CONTINUOUS_COLUMNS = [
        'age', 'capital_gain', 'capital_loss', 'hours_per_week', 'education_num'
    ]
    COLUMNS = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'gender',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
        'income_bracket'
    ]
    LABEL_COLUMN = ['income_bracket']
    PROTECTED_GROUPS = [
        'gender_Female', 'gender_Male', 'race_White', 'race_Black'
    ]
    CONTINUOUS_flag = [1, 0, 0, 0, 0]
    CONTINUOUS_bins = [4,
                       [-1, 1, 4000, 10000, 100000],
                       [-1, 1, 1800, 1950, 4500],
                       [0, 39, 41, 50, 100],
                       [0, 8, 9, 11, 16]]
    CC = list(zip(CONTINUOUS_COLUMNS, CONTINUOUS_flag, CONTINUOUS_bins))
    SHIFT = [['income_bracket', lambda x: 1 if x.strip().strip('.') == '>50K' else 0]]

    return load(path, cat=CATEGORICAL_COLUMNS, con=CC,
                raw=COLUMNS, y=LABEL_COLUMN, A=PROTECTED_GROUPS,
                shift=SHIFT)


def load(path, **kwargs):
    pd.options.mode.chained_assignment = None
    CATEGORICAL_COLUMNS = kwargs.get('cat', [])
    CONTINUOUS_COLUMNS = kwargs.get('con', [])
    COLUMNS = kwargs.get('raw', None)
    LABEL_COLUMN = kwargs.get('y')
    sensitive = kwargs.get('A')
    TRANSFORM_COLUMNS = kwargs.get('shift', [])

    CONTINUOUS_COLUMNS_name = [col for col, _, _ in CONTINUOUS_COLUMNS]

    df_raw = pd.read_csv(path, skipinitialspace=True, names=COLUMNS)


    if TRANSFORM_COLUMNS:
        for raw_column, func in TRANSFORM_COLUMNS:
            df_raw[raw_column] = df_raw[raw_column].apply(func)

    df = df_raw[CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS_name + LABEL_COLUMN]
    if CONTINUOUS_COLUMNS:
        for column_name, flag, rule in CONTINUOUS_COLUMNS:
            if flag:
                df[column_name] = pd.qcut(df[column_name], rule, labels=False)
            else:
                df[column_name] = pd.cut(df[column_name], rule, labels=False)

    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS_name)
    feature_names = list(df.keys())
    for v in LABEL_COLUMN:
        feature_names.remove(v)
    X = df[feature_names]
    y = df[LABEL_COLUMN[0]]
    S = df[sensitive]


    return X, y, S


def easy_load(path, **kwargs):
    pd.options.mode.chained_assignment = None
    CATEGORICAL_COLUMNS = kwargs.get('cat', [])
    CONTINUOUS_COLUMNS = kwargs.get('con', [])
    CONTINUOUS_split = kwargs.get('csp', [])
    TRANSFORM = kwargs.get('shift', [])
    continuous_dic = kwargs.get('cod', {})

    read_columns = kwargs.get('rc', None)
    if continuous_dic:
        CONTINUOUS_COLUMNS = list(continuous_dic)
    else:
        if CONTINUOUS_COLUMNS:
            continuous_dic = {c: r for c, r in zip(CONTINUOUS_COLUMNS, CONTINUOUS_split)}
    df_raw = pd.read_csv(path, skipinitialspace=True, names=read_columns)

    if TRANSFORM:
        for raw_column, func in TRANSFORM:
            df_raw[raw_column] = df_raw[raw_column].apply(func)

    df = df_raw

    if continuous_dic:
        for column_name in continuous_dic:
            rule = continuous_dic[column_name]
            if type(rule) == int and rule < 0:
                # qcut, 保证划分频率一样
                df[column_name] = pd.qcut(df[column_name], -rule, labels=False)
            else:
                df[column_name] = pd.cut(df[column_name], rule, labels=False)

    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS)
    return df

if __name__ == '__main__':
    # ['age_cat', 'gender', 'is_violent_recid', 'juv_fel_count', 'is_recid', 'priors_count',
    #                           'charge_degree', 'high_risk', 'race']
    dt = load_compas('data/to-ml-compas/compas_test.csv')
    print(dt[0].columns)
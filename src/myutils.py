import os
import pandas as pd

def check_create_dir(dir):

    if os.path.exists(dir):
        print(f'{dir} exists!')
    else:
        os.makedirs(dir, exist_ok = True)
        print(f'create {dir}!')

    return


def one_hot(df, cat_feats):
    """Return dataframe after one-hot encoding
    Args
        df        -- the dataframe containing data
        cat_feats -- the categorical features
    """

    for feat in cat_feats:
        one_hot = pd.get_dummies(df[feat], prefix = feat)
        df = df.drop(columns = [feat])
        df = df.merge(one_hot, left_index = True, right_index = True)

    return df

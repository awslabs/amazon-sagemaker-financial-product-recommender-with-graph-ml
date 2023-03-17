
import json
import numpy as np
import pandas as pd

USER = 'investor'
ITEM = 'security'

def summary(df):
    s = pd.DataFrame()
    dfn = df.select_dtypes(include=np.number)    
    s['count'] = df.count()
    s['nulls'] = df.isnull().sum()
    s['unique'] = df.nunique()
    s['type'] = df.dtypes
    s['mode'] = df.mode(axis=0).iloc[0]
    s['median'] = dfn.median()
    s['mean'] = dfn.mean()
    s['min'] = dfn.min()
    s['max'] = dfn.max()
    return s


def read_dataset(data_path):
    print(f'Reading {data_path}...')
    df = pd.read_csv(data_path, parse_dates=['date'], sep='\t')
    df.security += '*'  # to avoid collisions between investor and security names
    print(f'Dataset has {len(df):,} records and {df.shape[1]} fields')
    return df   

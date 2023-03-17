import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from utils import USER, ITEM


def stack(m, c, name):
    m.index.name = c + '_1'
    m.columns.name = c + '_2'
    ms = m.stack()
    ms.name = name
    return ms.reset_index()

    
def compute_matrices(train_df, e_names, e):
    m_dict = {
        'common': {},
        'euclidean': {}
    }
    for col in USER, ITEM:
        notc = USER if col == ITEM else ITEM
        #trades = train_df.assign(trade=1).pivot(index=col, columns=notc, values='trade').fillna(0)
        trades = train_df.assign(trade=1)[[USER, ITEM, 'trade']].drop_duplicates().pivot(index=col, columns=notc, values='trade').fillna(0)
        tradesT = trades.T.copy() # copy so we can have different names for index and columns
        m_dict['common'][col] = stack(trades.dot(tradesT), col, f'common_{notc}')
    for col in USER, ITEM:
        names = set(train_df[col])
        fil = e_names.isin(names)
        ec = e[fil, :]
        d = cdist(ec, ec)
        cols = e_names[fil].to_list()
        m_dict['euclidean'][col] = stack(pd.DataFrame(d, index=cols, columns=cols), col, 'distance')
    return m_dict


def build_explainability_df(reco_df, train_df, user_distances):
    # join recommendations and train on item to get all investors who traded the security for each reco
    merged1 = pd.merge(reco_df[[USER, ITEM, 'score']], train_df[[USER, ITEM]], 'left', left_on=ITEM, right_on=ITEM, suffixes=('_1', '_2'))
    # join with distance between investors and sort by distance
    merged2 = pd.merge(merged1, user_distances[user_distances.distance > 0], 'left').sort_values('distance')
    # get first (smallest distance)
    final = merged2.groupby([USER + '_1', ITEM], as_index=False)[USER + '_2'].first()
    # respect naming convention so we can aggregate later
    final.columns = [USER, ITEM, 'score']
    return final
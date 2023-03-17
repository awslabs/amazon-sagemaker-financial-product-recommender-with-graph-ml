#!/usr/bin/env python
# coding: utf-8


import os
import glob
import json
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler

from utils import read_dataset, USER, ITEM
from explain import compute_matrices, build_explainability_df


def os_system(cmd):
    print(cmd)
    os.system(cmd)


def ranked_list(li):
    return [(item, i) for i, item in enumerate(li)]


def rank(preds_list, true_set):
    for i, pred in enumerate(preds_list):
        if pred in true_set:
            return i + 1
    return 100_000


def compute_metrics(reco_dict, true_dict):
    support = set(reco_dict.keys()) & set(true_dict.keys())
    rank_dict = {user: rank(reco_dict[user], true_dict[user]) for user in support}
    r = {
            'mr': np.median(list(rank_dict.values())),
            'mrr': np.mean([1 / r for r in rank_dict.values()]),
    }
    for k in [1, 3, 10, 25, 50]:
        r[f'hit@{k}'] = np.mean([r <= k for r in rank_dict.values()])
    return r


def print_metrics(path, reco_dict, true_dict, meta_dict):
    print(path)
    r = compute_metrics(reco_dict, true_dict)
    for k, v in r.items():
        print(f'{k:<8}: {v:.3f}')
    r.update(meta_dict)
    with open(path, 'w') as f:
        json.dump(r, f)


def dict_diff(reco_dict, train_dict):
    d = {}
    for user, items in reco_dict.items():
        d[user] = [item for item in items if item not in train_dict[user]]
    return d


def train_predict(datapath, df, d1, d2, mode, gpu):
    """
    Train on [d1, d2[ and predict for d2
    """

    full_mode = mode == 'full'
    
    i_train = (d1 <= df.date) & (df.date < d2)
    i_test = df.date == d2
    
    recency_quantile = 0.1
    keep_last = 100

    model_dir = 'models'
    dataset_dir = 'datasets'
    

    # Generate the main graph data

    d1str = str(d1)[:10]
    d2str = str(d2)[:10]
    data_slice = os.path.basename(datapath).split('.')[0] + '_' + d1str + '_' + d2str
    print(data_slice)
    dset_dir = f'{dataset_dir}/{data_slice}'
    if full_mode:
        os_system(f'rm -rf {dset_dir}')
        os_system(f'mkdir -p {dset_dir}')

    df['timestamp'] = df.date.view(int) / 1e9 / 86400

    df_train = df[i_train].sort_values('timestamp', ascending=False).copy()
    rank_map = df_train[[USER, ITEM]].drop_duplicates().groupby(USER)[ITEM].apply(ranked_list).to_dict()
    rank_map = {k: dict(v) for k, v in rank_map.items()}
    df_train['rank'] = df_train.apply(lambda row: rank_map[row[USER]][row[ITEM]], axis=1)
    df_train['age'] = df_train.timestamp.max() - df_train.timestamp
    recency_threshold = np.quantile(df_train.age, recency_quantile)
    df_train['recency'] = df_train['age'].apply(lambda r: 'recent' if r <= recency_threshold else 'old')
    df_train['relation'] = df_train.recency
    df_train_final = df_train[df_train['rank'] < keep_last]
    df_train_final = df_train_final[[USER, 'relation', ITEM]].drop_duplicates()
    if full_mode:
        df_train_final[[USER, 'relation', ITEM]].to_csv(f'{dset_dir}/train.tsv', header=None, index=None, sep="\t")
    df_test = df.loc[i_test, [USER, ITEM]].drop_duplicates()
    df_test['relation'] = 'recent'
    if full_mode:
        df_test[[USER, 'relation', ITEM]].to_csv(f'{dset_dir}/test.tsv', header=None, index=None, sep="\t")

    # Train the model

    model_name = 'TransE_l2'
    lr = 0.25
    batch_size = 1024
    hidden_dim = 512
    max_step = 5000
    regularization_coef = 1e-8
    gamma = 19.9
    
    topk = 200
    
    model_path = f'{model_dir}/{model_name}_{data_slice}_0'
    if full_mode:
        os_system(f'rm -rf {model_path}')
        os_system(f'DGLBACKEND=pytorch dglke_train --model_name {model_name} --dataset {data_slice} --data_path ./{dset_dir}/ --format raw_udd_hrt --data_files train.tsv --lr {lr} --batch_size {batch_size} --hidden_dim {hidden_dim} --max_step {max_step} --log_interval 200 --regularization_coef {regularization_coef} --gpu {gpu} --gamma {gamma} --save_path {model_dir}')

        # Prepare inference data

        df_train_final[USER].drop_duplicates().to_csv(f'{dset_dir}/head.list', index=False, header=None)
        os_system(f'echo recent > {dset_dir}/rel.list')
        df_train_final[ITEM].drop_duplicates().to_csv(f'{dset_dir}/tail.list', index=False, header=None)

        # Make link predictions

        t = time.time()
        os_system(f"DGLBACKEND=pytorch dglke_predict --model_path {model_path} --exec_mode batch_head --format 'h_r_t' --data_files {dset_dir}/head.list {dset_dir}/rel.list {dset_dir}/tail.list --topK {topk} --raw_data --entity_mfile {dset_dir}/entities.tsv --rel_mfile {dset_dir}/relations.tsv --output {model_path}/reco.tsv")
        print(f'Inference took {time.time() - t:.1f}s')
    
    # Scoring model recommendations
    # Dependencies : reco.tsv df_test df_train df i_train i_test

    reco_df = pd.read_csv(f'{model_path}/reco.tsv', delimiter='\t').rename({'head': USER, 'tail': ITEM}, axis=1)
    true_df = df_test.copy()
    train_dict = df_train.groupby(USER)[ITEM].apply(set).to_dict()
    reco_dict = reco_df.groupby(USER)[ITEM].apply(list).to_dict()
    true_dict = true_df.groupby(USER)[ITEM].apply(set).to_dict()
    print_metrics(f'{model_path}/metrics_mod_nat.json', reco_dict, true_dict, {'date': d2str, 'model': model_name, 'filter': 'none'})
    print_metrics(f'{model_path}/metrics_mod_new.json', dict_diff(reco_dict, train_dict), true_dict, {'date': d2str, 'model': model_name, 'filter': 'new'})

    # Scoring popular items as a benchmark
    
    pop_items = list(df_train[ITEM].value_counts().index)
    pop_recos = {user: pop_items for user in reco_dict.keys()}
    print_metrics(f'{model_path}/metrics_pop_nat.json', pop_recos, true_dict, {'date': d2str, 'model': 'most popular', 'filter': 'none'})
    print_metrics(f'{model_path}/metrics_pop_new.json', dict_diff(pop_recos, train_dict), true_dict, {'date': d2str, 'model': 'most popular', 'filter': 'new'})

    # Aggregate metric json files into one CSV
    
    pd.concat([pd.read_json(path, lines=True) for path in glob.glob(f'{model_path}/metrics_*.json')]).to_csv(f'{model_path}/metrics.csv', index=False)

    # Generate recommendation file

    cols = ['prediction_date', 'type', USER, ITEM, 'score']

    print('Preparing recommendation file')
    print('Scores...')
    reco_df['prediction_date'] = d2
    reco_df['type'] = 'score'
    reco_df.score = MinMaxScaler().fit_transform(reco_df.score.values.reshape(-1, 1))
    
    print('Past and current trades...')
    qs = df[i_train|i_test].copy()
    qs['prediction_date'] = d2
    qs.loc[i_train, 'type'] = 'train'
    qs.loc[i_test, 'type'] = 'test'
    qs['score'] = 1
   
    e_names = pd.read_csv(f'{dset_dir}/entities.tsv', sep='\t', header=None, names=['index', 'entity'])['entity']
    e = np.load(f'{model_path}/{data_slice}_{model_name}_entity.npy')
    print('Distance matrices...')
    m_dict = compute_matrices(df_train_final, e_names, e)

    if False:
        print('Explanations...')
        explain = build_explainability_df(reco_df, df_train_final, m_dict['euclidean'][USER])
        explain['prediction_date'] = d2
        explain['type'] = 'explanation'

        print('Final file...')
        t = pd.concat([reco_df[cols], qs[cols], explain[cols]]).pivot(index=['prediction_date', USER, ITEM], columns='type', values='score')
        t['rank'] = t.groupby(['prediction_date', USER])['score'].rank(ascending=False)
        t['train test score rank explanation'.split()].to_csv(f'{model_path}/qs_reco.tsv', sep='\t')

    if True:
        print('Dump distance files...')
        for c in USER, ITEM:
            ddf = pd.merge(m_dict['euclidean'][c], m_dict['common'][c]).sort_values([c + '_1', 'distance'])
            #ddf['prediction_date'] = d2
            tag = 'user' if c == USER else 'item'
            #ddf.to_csv(f'{model_path}/distances_{tag}.tsv', sep='\t')
            ddf.to_pickle(f'{model_path}/distances_{tag}.pkl')
            
    for src, dst_env in [(model_path, 'SM_MODEL_DIR'), (dset_dir, 'SM_OUTPUT_DATA_DIR')]:
        dst = os.environ.get(dst_env)
        if dst:
            os_system(f'mv {src} {dst}')
            
    print('Done')

    
if __name__ == "__main__":
    
    print('>>>>>>>>>>>>>>>>>>>>>>>>')
    os_system('pwd')
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default='trades.tsv', help="input file name")
    parser.add_argument("--training-index",   type=int, default=71, help="start of training index")
    parser.add_argument("--prediction-index", type=int, default=73, help="prediction index")
    parser.add_argument("--mode", type=str, default='full', help="full|post-process")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu index, -1 for cpu")
    
    args = parser.parse_args()

    for k in "SM_CHANNEL_TRAINING".split():
        v = os.environ.get(k)
        print(k, ':', v)
        if v:
            os_system(f'ls {v}')
    
    datapath = os.path.join(os.environ.get('SM_CHANNEL_TRAINING', default=''), args.datapath)
        
    df = read_dataset(datapath)    
    dates = sorted(df.date.unique())
    train_predict(args.datapath, df, dates[args.training_index], dates[args.prediction_index], args.mode, args.gpu)


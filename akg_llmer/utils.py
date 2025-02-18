import pickle
import numpy as np
import pandas as pd
import json
from serialization import *


def get_labels(path='./d10_ImDb.pkl', num_neg=100):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if 'k' in data:
        num_neg = data['k']

    # pair each fact with #num_neg negatives for ranking metrics
    k_pair = []
    for i in range(data['pos'].shape[0]):
        indices = data['pos'][i].indices
        if len(indices) > 0:
            k_pair += [[i, j]+data['neg'][i][:num_neg] for j in indices]
    k_pair = np.array(k_pair)
    return k_pair[:, 0], k_pair[:, 1:]


def load_entity(config_data):
    root_path = config_data['root_path']
    dataset = config_data['dataset']
    dsource = config_data['source']
    dtarget = config_data['target']

    # Data Preprocessing
    sep = config_data['sep'] if 'sep' in config_data else ','
    if config_data['name'] == 'd2':
        df_l = pd.read_csv(
            f"{root_path}/{dataset}/{dsource}.csv", sep=sep).drop('id', axis=1)
        df_r = pd.read_csv(f"{root_path}/{dataset}/{dtarget}.csv", sep=sep)
    else:
        df_l = pd.read_csv(
            f"{root_path}/{dataset}/{dsource}.csv", index_col=0, sep=sep)
        df_r = pd.read_csv(
            f"{root_path}/{dataset}/{dtarget}.csv", index_col=0, sep=sep)
    if 'col_l' in config_data:
        df_l.columns = config_data['col_l']
        df_r.columns = config_data['col_r']
    if 'aggregate value' in df_l.columns:
        df_l.drop('aggregate value', axis=1, inplace=True)
        df_r.drop('aggregate value', axis=1, inplace=True)
    return df_l, df_r


def serial_entity(config_data, stype: str = 'fixed', indices=None):
    df_l, df_r = load_entity(config_data)
    file_path = f"{config_data['root_path']}/{config_data['dataset']}"

    if stype == 'json':
        with open(f"{file_path}/{config_data['source']}.json", 'r') as f:
            # Load the data from the file
            left = json.load(f)
        with open(f"{file_path}/{config_data['target']}.json", 'r') as f:
            # Load the data from the file
            right = json.load(f)
        if indices is not None:
            seq_l = [json.dumps(left[row]) for row in indices.ltable_id.values]
            seq_r = [json.dumps(right[row])
                     for row in indices.rtable_id.values]
        else:
            return left, right
    elif stype == 'pairwise':
        print('Encode attributes pairwisely!')
        if indices is not None:
            seq_l = csv_pairwise(df_l.loc[indices.ltable_id.values].reset_index(
                drop=True), df_r.loc[indices.rtable_id.values].reset_index(drop=True))
            seq_r = [''] * len(seq_l)
        else:
            raise NotImplementedError
    elif stype == 'span':
        if 'injector' in config_data:
            injector = config_data['injector']
            if indices is not None:
                seq_l = [injector.transform(e) for e in csv_fixed(
                    df_l.loc[indices.ltable_id.values])]
                seq_r = [injector.transform(e) for e in csv_fixed(
                    df_r.loc[indices.rtable_id.values])]
            else:
                seq_l = [injector.transform(e) for e in csv_fixed(df_l)]
                seq_r = [injector.transform(e) for e in csv_fixed(df_r)]
        else:
            serial_func = eval(f"csv_{stype}")
            if indices is not None:
                seq_l = serial_func(
                    df_l.loc[indices.ltable_id.values], config_data['span'])
                seq_r = serial_func(
                    df_r.loc[indices.rtable_id.values], config_data['span'])
            else:
                seq_l = serial_func(df_l, config_data['span'])
                seq_r = serial_func(df_r, config_data['span'])
    else:
        serial_func = eval(f"csv_{stype}")
        if indices is not None:
            seq_l = serial_func(df_l.loc[indices.ltable_id.values])
            seq_r = serial_func(df_r.loc[indices.rtable_id.values])
        else:
            seq_l = serial_func(df_l)
            seq_r = serial_func(df_r)
    return seq_l, seq_r

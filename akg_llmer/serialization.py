import re
import numpy as np
from collections import Counter


def csv_fixed(source):
    # fixed order with special separators [COL] [VAL]
    # input pandas DataFrame
    # output list
    entity_list = []
    for _, row in source.iterrows():
        entity_list.append(
            ' '.join([f'[COL] {key} [VAL] {row[key]}' for key in source.columns]))
    return entity_list


def csv_random(source):
    # random shuffled order with special separators [COL] [VAL]
    entity_list = []
    for _, row in source.iterrows():
        entity_list.append(' '.join(
            [f'[COL] {key} [VAL] {row[key]}' for key in np.random.permutation(source.columns)]))
    return entity_list


def csv_valid(source):
    # fixed order with special separators [COL] [VAL] but only keeps valid / non-NaN pairs
    entity_list = []
    for _, row in source.iterrows():
        entity_list.append(' '.join(
            [f'[COL] {key} [VAL] {row[key]}' for key in source.columns if not row[key] in [None, np.nan]]))
    return entity_list


def csv_plain(source):
    # fixed order without special separators and only keeps valid / non-NaN pairs
    entity_list = []
    for _, row in source.iterrows():
        entity_list.append(' '.join(
            [f'{key} {row[key]}' for key in source.columns if row[key] is not None]))
    return entity_list


def csv_span(source, span_key=None):
    # fixed order with special separators [COL] [VAL] and add special tokens </s> around the target pairs
    # e.g. if the target column is SongName then "</s> [COL] Song_Name [VAL] Lips Are Movin </s>"
    entity_list = []
    for _, row in source.iterrows():
        tmp = []
        for key in source.columns:
            if key in span_key:
                tmp.append(f'</s> [COL] {key} [VAL] {row[key]} </s>')
            else:
                tmp.append(f'[COL] {key} [VAL] {row[key]}')
        entity_list.append(' '.join(tmp))
    return entity_list


def csv_pairwise(source, target):
    # serialize entities pairwisely with aligned features
    # e.g. "[COL] Song_Name [VAL1] Runaway Train Cam Country [VAL2] Blue On Black"
    entity_list = []
    combined = Counter(source.columns.tolist() + target.columns.tolist())
    cols = list(
        dict(sorted(combined.items(), key=lambda item: item[1], reverse=True)).keys())

    for i in range(len(source)):
        r_source = source.loc[i]
        r_target = target.loc[i]
        tmp = []
        for c in cols:
            val1 = r_source[c] if c in r_source else 'nan'
            val2 = r_target[c] if c in r_target else 'nan'
            tmp.append(f'[COL] {c} [VAL1] {val1} [VAL2] {val2}')
        entity_list.append(' '.join(tmp))
    return entity_list


def csv_pairwisex(source, target):
    # serialize entities pairwisely with aligned features with repeated column name
    # e.g. "[COL] Song_Name [VAL] Runaway Train Cam Country [COL] Song_Name [VAL] Blue On Black"
    entity_list = []
    combined = Counter(source.columns.tolist() + target.columns.tolist())
    cols = list(
        dict(sorted(combined.items(), key=lambda item: item[1], reverse=True)).keys())

    for i in range(len(source)):
        r_source = source.loc[i]
        r_target = target.loc[i]
        tmp = []
        for c in cols:
            val1 = r_source[c] if c in r_source else 'nan'
            val2 = r_target[c] if c in r_target else 'nan'
            tmp.append(f'[COL] {c} [VAL] {val1} [COL] {c} [VAL] {val2}')
        entity_list.append(' '.join(tmp))
    return entity_list


def csv_walk(source, num_walk=3):
    # serialize entities without special separators and downsampled (with replacement) features
    entity_list = []
    cols = source.columns
    num_walk = min(num_walk, len(cols)-1)
    for _, row in source.iterrows():
        e = row[cols[0]]
        entity_list.append(' '.join([f'{e} {key} {row[key]}' for key in np.random.choice(
            cols[1:], num_walk) if not row[key] in [None, np.nan]]))
    return entity_list


def json_fixed(source):
    # serialize JSON object, deprecated; replaced by json.dumps()
    entity_list = []
    for e in source:
        tmp = []
        for key, val in e.items():
            if key != 'id':
                if type(val) in {int, float, str}:
                    tmp.append(f"[COL] {key} [VAL] {val} ")
                else:
                    if type(val) == list:
                        tmp.append(f"[COL] {key} [VAL] {' '.join(val)} ")
                    elif type(val) == dict:
                        tmp.append(f"[COL] {key} [VAL] {' '.join([f'[COL] {k} [VAL] {v}' for k, v in val.items()])} ")
        entity_list.append(re.sub(r'\s+', ' ', ''.join(tmp).rstrip()))
    return entity_list

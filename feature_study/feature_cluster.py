from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from pkg_resources import resource_filename
from sklearn.cluster import k_means


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) > 0)


if __name__ == '__main__':
    config = get_config('/../config/config.yaml')
    train_path = resource_filename(__name__, config['train']['path'])
    dev_path = resource_filename(__name__, config['dev']['path'])
    test_path = resource_filename(__name__, config['test']['path'])
    df_train = pd.read_json(train_path)
    df_dev = pd.read_json(dev_path)
    df_test = pd.read_json(test_path)
    n = 25
    m = len(df_train.index)
    matrix = np.zeros((n, m))

    for index, row in df_train.iterrows():
        temp = row['feature_input'][:-1]
        feature_vec = temp[1:]
        feature_ls = list(feature_vec.split(","))
        for idx, i in enumerate(feature_ls):
            feat = float(i)
            matrix[idx][index] = feat
    selected_matrix = np.delete(matrix, [5, 6, 16, 23], axis=0)
    clustering = k_means(selected_matrix, n_clusters=3)
    label = clustering[1]
    print(label)
    feature_names = ['num_chars', 'num_words', 'positive', 'negative', 'nouns', 'modal', 'adjectives', 'arousal',
                     'fear', 'pop', 'manip', 'scandal', 'numbers', 'ner', 'oov', 'question', 'exag', 'quatation',
                     'personal', 'hate', 'negations', 'title_num_words', 'title_question', 'title_exclam', 'title_cap']

    selected_feature_names = ['num_chars', 'num_words', 'positive', 'negative', 'nouns', 'arousal',
                              'fear', 'pop', 'manip', 'scandal', 'numbers', 'ner', 'oov', 'question', 'quatation',
                              'personal', 'hate', 'negations', 'title_num_words', 'title_question', 'title_cap']

    for dup in sorted(list_duplicates(label)):
        print(dup)
        cluster = dup[0]
        print(cluster)
        occur = dup[1]
        print(occur)
        ls_name = []
        for _ in occur:
            name = selected_feature_names[_]
            ls_name.append(name)
        print(ls_name)

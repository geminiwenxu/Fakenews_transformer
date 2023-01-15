import os

import pandas as pd
import yaml
from pkg_resources import resource_filename
from sklearn.model_selection import train_test_split

from data_scraping.get_data_statistics import get_source


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def get_source_data(dict_source_groups, source):
    """
    get a dataset from chosen data sources with same proportion of true and false news
    :param dict_source_groups:
    :param source: chosen websites in a source group
    :return: dataset by source group
    """
    df_true_source = df_true[df_true['source'].isin(dict_source_groups[source])]
    df_fake = df_source[df_source['label_id'] == 0]
    # the proportion false/true should be the same for comparison
    int_fake = int(df_true_source.shape[0] * 0.38)
    df_bild = pd.concat([df_fake.sample(n=int_fake, replace=False, random_state=42), df_true_source]
                        )

    news_train, news_test, = train_test_split(df_bild, test_size=0.2, shuffle=True)
    news_test, news_dev, = train_test_split(news_test, test_size=0.5, shuffle=True)

    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, source)
    os.makedirs(final_directory)
    news_train.to_json(path_or_buf=source + '/feats_train_norm.json', force_ascii=False, orient='records')
    news_test.to_json(path_or_buf=source + '/feats_test_norm.json', force_ascii=False, orient='records')
    news_dev.to_json(path_or_buf=source + '/feats_dev_norm.json', force_ascii=False, orient='records')


config = get_config('/../config/config.yaml')
test_path = resource_filename(__name__, config['test']['path'])
df_test = pd.read_json(test_path)
test_path = resource_filename(__name__, config['dev']['path'])
df_dev = pd.read_json(test_path)
test_path = resource_filename(__name__, config['train']['path'])
df_train = pd.read_json(test_path)

df_all = pd.concat([df_test, df_dev, df_train])

df_source = get_source(df_all)

df_true = df_source[df_source['label_id'] == 1]

dict_source_groups = {'bild': ['bild'],
                      'tz': ['tagesspiegel', 'zeit', 'sueddeutsche'],
                      'leicht': ['nachrichtenleicht', 'ndr', 'mdr', 'sr', 'kleinezeitung'],
                      'vp': ['volksverpetzer']
                      }

get_source_data(dict_source_groups, 'bild')
get_source_data(dict_source_groups, 'tz')
get_source_data(dict_source_groups, 'leicht')
get_source_data(dict_source_groups, 'vp')

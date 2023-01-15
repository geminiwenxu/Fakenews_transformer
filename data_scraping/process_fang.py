import json
from os import listdir
from os.path import join

import pandas as pd


def merge_json(filename):
    result = list()
    for f1 in filename:
        with open(f1, 'r') as infile:
            text = json.load(infile)
            result.append(text)

    with open('/data_scraping/fang.json', 'w') as output_file:
        json.dump(result, output_file)


files = [join('/Users/darialinke/PycharmProjects/fang-covid/articles/', f) for f in
         listdir('/Users/darialinke/PycharmProjects/fang-covid/articles/')]
merge_json(files)
df_fang = pd.read_json('/Users/darialinke/PycharmProjects/Fakenews_transformer/data_scraping/fang.json')
df_fang = df_fang.drop(['twitter-history', 'source'], axis=1)
df_true = df_fang[df_fang['label'] == 'real']
df_true['label_id'] = '1'
df_fake = df_fang[df_fang['label'] == 'fake']
df_fake['label_id'] = '0'

# real:28056
# fake 13186
# 1:1 matching
df_true = df_true.sample(n=13186).reset_index(drop=True)
df_all = pd.concat([df_true, df_fake])
df_all = df_all.rename(columns={"header": "title", "article": "text"})
df_all[['url', 'date', 'title', 'text', 'label', 'label_id']].to_json(
    path_or_buf='../data_scraping/fang_processed.json', force_ascii=False, orient='records')

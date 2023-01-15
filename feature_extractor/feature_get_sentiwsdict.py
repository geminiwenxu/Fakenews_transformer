import pickle
import re

import numpy as np
import pandas as pd
import spacy


def get_unique_words(text):
    nlp = spacy.load('de_core_news_sm')
    doc = nlp(text)
    token = []
    for t in doc:
        if t.is_stop is False and t.is_space is False and t.text.__len__() > 3:
            token.append(t.lemma_.lower())
    arr = np.unique(token, return_counts=True, return_index=False)
    df = pd.DataFrame({'token': arr[0], 'count': arr[1]})
    return df


def comparison(type_news):
    """
    comparison of used words in true news and fake news
    :param type_news:
    :return:
    """
    if type_news == 'fake':
        news = pd.read_json('/Users/darialinke/PycharmProjects/Fakenews_transformer/Data Scraping/fake_news.json')
    else:
        news = pd.read_json('/Users/darialinke/PycharmProjects/Fakenews_transformer/Data Scraping/true_news.json')
    text = ''.join(str(x) for x in news['article'].values.tolist())

    pickle.dump(get_unique_words(text.replace('\t', '').replace("\xad", '')), open(type_news + ".p", "wb"))


def get_dict_from_SentiWs(type):
    """
    converts SentiWs file to dict with emotion values and contains all word forms
    :param type: positive or negative
    :return: pickled dict
    """
    df_raw = pd.read_csv('./SentiWS_v2/SentiWS_v2.0_' + type + '.txt', sep='\t', header=None,
                         names=['word', 'value', 'flek'])
    df_raw['word'] = df_raw['word'].apply(lambda x: re.sub('\|[A-Z]+', '', x))
    df_lexem = df_raw.drop(['flek'], axis=1)

    df_all = df_raw.flek.str.split(',').apply(pd.Series)
    df_all.index = df_raw.word
    df_all = df_all.stack().reset_index('word')
    df_all = df_all.merge(df_raw, how='left', on='word')
    df_all.drop(['flek', 'word'], axis=1, inplace=True)
    df_all.rename(columns={0: 'word'}, inplace=True)

    # stack both dfs
    df_final = pd.concat([df_all, df_lexem])
    df_final['word'] = df_final['word'].str.lower()
    dict_all = df_final.set_index('word').to_dict()['value']

    pickle.dump(dict_all, open('dict' + type + '.p', "wb"))


if __name__ == "__main__":
    # Extraction from SentiWs
    # process the SentiWs file to dictioanary for emotional value look up
    get_dict_from_SentiWs('Positive')
    get_dict_from_SentiWs('Negative')

import pandas as pd
import pickle

import pandas as pd
import spacy
import yaml
from pkg_resources import resource_filename
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from transformers import pipeline

from feature_extractor.feats_class import Article, Headline


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def main():
    config = get_config('/../config/config.yaml')
    df = pd.read_json(open(resource_filename(__name__, config['data_all']['path'])))

    nlp = spacy.load('de_core_news_lg')
    dict_positive = pickle.load(open(resource_filename(__name__, config['path_dict_pos']['path']), "rb"))
    dict_negative = pickle.load(open(resource_filename(__name__, config['path_dict_neg']['path']), "rb"))
    pd_arousal = pd.read_csv(filepath_or_buffer=resource_filename(__name__, config['path_arousal']['path']), sep=';')

    dict_arousal = pd_arousal.set_index('WORD_LOWER').to_dict()['AROUSAL_MEAN']

    # dicts
    topoi = pd.read_csv(resource_filename(__name__, config['path_dict_topoi']['path']), sep=';')
    topoi = topoi.term.str.lower().tolist()

    anto = pd.read_csv(resource_filename(__name__, config['path_dict_anto']['path']), sep=';')
    anto = anto.term.str.lower().tolist()

    style = pd.read_csv(resource_filename(__name__, config['path_dict_style']['path']), sep=';')
    style = style.term.str.lower().tolist()

    # for hatespeech detection
    model_name = 'EIStakovskii/german_toxicity_classifier_plus_v2'
    toxicity_pipeline = pipeline('text-classification', model=model_name, tokenizer=model_name)

    df_new = df[~df.index.duplicated(keep='first')]

    list_feats = []
    count = 1
    for index, row in df_new.iterrows():
        feats_article = Article(raw_text=row['text'],
                                dict_positive=dict_positive, dict_negative=dict_negative, dict_style=style,
                                dict_anto=anto, dict_topo=topoi,
                                dict_arousal=dict_arousal, toxicity_pipeline=toxicity_pipeline, nlp=nlp
                                )

        feats_headline = Headline(row['title'], nlp)

        feats_all = feats_article.return_results() + feats_headline.return_results()

        list_feats.append(feats_all)
        count += 1

    df_feats = pd.DataFrame(list_feats)

    scaler = preprocessing.RobustScaler(quantile_range=(5.0, 95.0))
    df_feats_scaled = scaler.fit_transform(df_feats)
    df_feats_scaled = pd.DataFrame(df_feats_scaled)

    df_new['feature_input'] = df_feats_scaled.values.tolist()
    df_new['feature_input'] = df_new['feature_input'].astype(str)

    news_train, news_test, = train_test_split(df_new, test_size=0.2, shuffle=True)
    news_test, news_dev, = train_test_split(news_test, test_size=0.5, shuffle=True)

    news_train.to_json(path_or_buf='feats_train_norm.json', force_ascii=False, orient='records')
    news_test.to_json(path_or_buf='feats_test_norm.json', force_ascii=False, orient='records')
    news_dev.to_json(path_or_buf='feats_dev_norm.json', force_ascii=False, orient='records')

    # save the raw feats before normalization

    df_feats['label_id'] = df_new['label_id']
    df_feats['url'] = df_new['url']  # act as a key for later join
    df_feats.to_csv('feats_raw.csv')


if __name__ == "__main__":
    main()

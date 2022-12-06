import json
import pandas as pd
import yaml
from pkg_resources import resource_filename
from transformers import AutoModel, pipeline
from draft_feats import Article, Headline
import pickle
import spacy
from sklearn import preprocessing
import os


print("Start")

def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def main():
    config = get_config('/../config/config.yaml')
    train_path = resource_filename(__name__, config['dev']['path'])
    df = pd.read_json(train_path)

    nlp = spacy.load('de_core_news_lg')
    # get dictionaries for positive, negative and arousal words
    # load keyword list
    dict_positive = pickle.load(open("dictPositive.p", "rb"))
    dict_negative = pickle.load(open("dictNegative.p", "rb"))
    keywords = pd.read_csv('Keywords_fake.txt')
    keywords = keywords.keyword.str.lower().tolist()
    pd_arousal = pd.read_csv(filepath_or_buffer='list_arousal.csv', sep=';')
    dict_arousal = pd_arousal.set_index('WORD_LOWER').to_dict()['AROUSAL_MEAN']

    # for hatespeech detection
    model_name = 'ml6team/distilbert-base-german-cased-toxic-comments'
    toxicity_pipeline = pipeline('text-classification', model=model_name, tokenizer=model_name)


    df_new = df#.iloc[:5,:]

    list_feats = []
    for index, row in df_new.iterrows():
        feats_article = Article(raw_text=row['text'],
                                dict_positive=dict_positive, dict_negative=dict_negative, keywords=keywords,
                                dict_arousal=dict_arousal, toxicity_pipeline=toxicity_pipeline, nlp=nlp
                                )
        #print(feats_article.return_results())

        feats_headline = Headline(row['title'], nlp)
        #print(feats_headline.return_results())

        feats_all = feats_article.return_results() + feats_headline.return_results()

        list_feats.append(feats_all)


    #df_feats = pd.DataFrame(list_feats)
    min_max_scaler = preprocessing.MinMaxScaler()
    df_feats_scaled = min_max_scaler.fit_transform(list_feats)
    df_feats_scaled = pd.DataFrame(df_feats_scaled)
    df_new['feature_input'] = df_feats_scaled.values.tolist()
    df_new['feature_input'] = df_new['feature_input'].astype(str)


    df_new.to_json(path_or_buf='news_feats_dev_norm.json', force_ascii=False, orient='records')


if __name__ == "__main__":
    main()

from collections import Counter

import nltk
import pandas as pd
import spacy
import yaml
from nltk.tokenize import sent_tokenize
from pkg_resources import resource_filename

nltk.download('punkt')


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def num_words(df):
    nlp = spacy.load('de_core_news_sm')
    num_adj = 0
    num_words = 0
    ents = Counter()
    num_chars = 0
    for idx, row in df.iterrows():
        num_chars += len(row['text']) / len(row['text'].split())
        num_words += len(row['text'].split()) / len(sent_tokenize(row['text']))
        doc = nlp(row['text'])
        for token in doc:
            if token.pos_ == "ADJ":
                num_adj += 1
        for ent in doc.ents:
            ents[f"{ent.label_}:{ent.text}"] += 1
    print(num_words / idx)
    num_ner = sum(ents.values())
    print(num_ner / idx)
    print(num_adj / idx)
    print(num_chars / idx)


def len_title(df):
    train_path = resource_filename(__name__, config['train']['path'])
    dev_path = resource_filename(__name__, config['dev']['path'])
    test_path = resource_filename(__name__, config['test']['path'])
    df_train = pd.read_json(train_path)
    df_dev = pd.read_json(dev_path)
    df_test = pd.read_json(test_path)
    df_all_raw = pd.concat([df_test, df_dev, df_train])
    len_title = 0
    num_cap = 0
    for idx, row in df.iterrows():
        for i, r in df_all_raw.iterrows():
            if row['text'] == r['text']:
                len_title += len(r['title'].split())
                num_cap += sum(1 for c in r['title'] if c.isupper())
    print(len_title / idx)
    print(num_cap / idx)


if __name__ == '__main__':
    config = get_config('/../config/config.yaml')
    misclassification_path = resource_filename(__name__, config['misclassification']['path'])
    df_misclassification = pd.read_json(misclassification_path)
    # print(df_misclassification)
    correct_classification_path = resource_filename(__name__, config['correct_classification']['path'])
    df_correct_classification = pd.read_json(correct_classification_path)
    # print(df_correct_classification)
    num_words(df_misclassification)
    len_title(df_misclassification)
    print("------------------------------")
    num_words(df_correct_classification)
    len_title(df_correct_classification)

import nltk
import pandas as pd
import yaml
from pkg_resources import resource_filename

nltk.download('punkt')
from nltk.tokenize import sent_tokenize


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def num_words(df):
    num_words = 0
    num_sentence = 0
    num_cap = 0
    for idx, row in df.iterrows():
        num_words += len(row['text'].split())
        num_sentence += len(sent_tokenize(row['text']))
        num_cap += sum(1 for c in row['text'] if c.isupper())
    print(num_words / 51)
    print(num_sentence / 51)
    print(num_cap/51)


def len_title(df):
    train_path = resource_filename(__name__, config['train']['path'])
    dev_path = resource_filename(__name__, config['dev']['path'])
    test_path = resource_filename(__name__, config['test']['path'])
    df_train = pd.read_json(train_path)
    df_dev = pd.read_json(dev_path)
    df_test = pd.read_json(test_path)
    df_all_raw = pd.concat([df_test, df_dev, df_train])
    len_title = 0
    for idx, row in df.iterrows():
        for i, r in df_all_raw.iterrows():
            if row['text'] == r['text']:
                len_title += len(r['title'].split())
    print(len_title / 51)


if __name__ == '__main__':
    config = get_config('/../config/config.yaml')
    misclassification_path = resource_filename(__name__, config['misclassification']['path'])
    df_misclassification = pd.read_json(misclassification_path)
    correct_classification_path = resource_filename(__name__, config['correct_classification']['path'])
    df_correct_classification = pd.read_json(correct_classification_path)
    num_words(df_misclassification)
    print("------------------------------")
    num_words(df_correct_classification)



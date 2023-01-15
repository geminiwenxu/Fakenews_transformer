import pandas as pd
import spacy
import yaml
from pkg_resources import resource_filename
from sklearn.model_selection import train_test_split


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def get_wrong_lenghts(nlp):
    # get train & test data set
    df_wrong = pd.read_json(resource_filename(__name__, '/../analyze_results/wrong_Baseline.json'))
    df_wrong['doc'] = df_wrong['text'].apply(lambda x: nlp(x))
    df_wrong['len'] = df_wrong['doc'].apply(lambda x: len(x))
    print("Average lenght Wrong Base ", df_wrong['len'].mean())

    # """
    df_correct = pd.read_json(resource_filename(__name__, '/../analyze_results/correct_Baseline.json'))
    df_correct['doc'] = df_correct['text'].apply(lambda x: nlp(x))
    df_correct['len'] = df_correct['doc'].apply(lambda x: len(x))
    print("Average lenght Correct Base ", df_correct['len'].mean())

    df_wrong_fusion = pd.read_json(resource_filename(__name__, '/../analyze_results/wrong_Fusion.json'))
    df_wrong_fusion['doc'] = df_wrong_fusion['text'].apply(lambda x: nlp(x))
    df_wrong_fusion['len'] = df_wrong_fusion['doc'].apply(lambda x: len(x))
    print("Average lenght Fusion Base ", df_wrong_fusion['len'].mean())


def get_new_data_by_len(nlp):
    config = get_config('/../config/config.yaml')
    df_all = pd.read_json(open(resource_filename(__name__, config['data_all']['path'])))
    df_all['doc'] = df_all['text'].apply(lambda x: nlp(x))
    df_all['len'] = df_all['doc'].apply(lambda x: len(x))
    df_all = df_all[df_all['len'] > 900]

    df_all.drop(['doc', 'len'], axis=1, inplace=True)
    # make sure that the proportions of real/fake are same as in previous experiments
    df_true = df_all[df_all['label_id'] == 1]
    df_fake = df_all[df_all['label_id'] == 0]
    int_fake = int(df_true.shape[0] * 0.38)
    df_final = pd.concat([df_fake.sample(n=int_fake, replace=False, random_state=42), df_true]
                         )
    news_train, news_test, = train_test_split(df_final, test_size=0.2, shuffle=True)
    news_test, news_dev, = train_test_split(news_test, test_size=0.5, shuffle=True)
    news_train.to_json(path_or_buf='feats_train_norm.json', force_ascii=False, orient='records')
    news_test.to_json(path_or_buf='feats_test_norm.json', force_ascii=False, orient='records')
    news_dev.to_json(path_or_buf='feats_dev_norm.json', force_ascii=False, orient='records')


def main():
    nlp = spacy.load('de_core_news_lg')
    get_new_data_by_len(nlp)
    get_wrong_lenghts(nlp)


if __name__ == "__main__":
    main()

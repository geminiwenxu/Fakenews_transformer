import pandas as pd
import yaml
from pkg_resources import resource_filename
from sklearn.model_selection import train_test_split
from feature_selection import get_feats_from_col


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def main():
    print("get data")
    config = get_config('/../config/config.yaml')
    test_path = resource_filename(__name__, config['test']['path'])
    df_test = pd.read_json(test_path)
    test_path = resource_filename(__name__, config['dev']['path'])
    df_dev = pd.read_json(test_path)
    test_path = resource_filename(__name__, config['train']['path'])
    df_train = pd.read_json(test_path)

    # needed later to join again the feats
    df_all_raw = pd.concat([df_test, df_dev, df_train])
    df_final = df_all_raw.copy()
    df_final.drop('feature_input', axis=1, inplace=True)
    df_feats_disag = get_feats_from_col(df_all_raw)


    # groups of feats
    groups_feats = {'title': ['title_num_words', 'title_question', 'title_exclam', 'title_cap'],
                    'emotion': ['positive', 'negative', 'hate', 'arousal', 'exclam'],
                    'complexity': ['num_chars', 'num_words'],
                    'uncertainty': ['modal', 'numbers', 'question'],
                    'subjectivity': ['personal', 'quatation', 'negations', 'oov'],
                    'lexica': ['topoi', 'anto', 'style'],
                    'formal': ['nouns', 'adjectives', 'ner']
                    }


    ##### your ablation feature here
    df_feats_disag.drop('title_num_words', inplace=True, axis=1)

    #print(df_feats_chosen.columns)
    df_final['feature_input'] = df_feats_disag.iloc[:, 5:].values.tolist()  # all feats cols back to list
    df_final['feature_input'] = df_final['feature_input'].astype(str)

    news_train, news_test, = train_test_split(df_final, test_size=0.2, shuffle=True)
    news_test, news_dev, = train_test_split(news_test, test_size=0.5, shuffle=True)

    news_train.to_json(path_or_buf='feats_train_norm_title_num_words.json', force_ascii=False, orient='records')
    news_test.to_json(path_or_buf='feats_test_norm_title_num_words.json', force_ascii=False, orient='records')
    news_dev.to_json(path_or_buf='feats_dev_norm_title_num_words.json', force_ascii=False, orient='records')


if __name__ == "__main__":
    main()


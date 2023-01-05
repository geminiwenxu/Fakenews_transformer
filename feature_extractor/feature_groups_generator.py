import pandas as pd
import yaml
from pkg_resources import resource_filename
from sklearn.model_selection import train_test_split
<<<<<<< HEAD

=======
>>>>>>> 5937b8681549231a4fefa4a666f2296c5e25e942
from feature_selection import get_feats_from_col


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf

<<<<<<< HEAD

=======
>>>>>>> 5937b8681549231a4fefa4a666f2296c5e25e942
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

<<<<<<< HEAD
    # groups of feats
    groups_feats = {'title': ['title_cap'],
=======

    # groups of feats
    groups_feats = {'title': ['title_num_words', 'title_question', 'title_exclam', 'title_cap'],
>>>>>>> 5937b8681549231a4fefa4a666f2296c5e25e942
                    'emotion': ['positive', 'negative', 'fear', 'hate', 'arousal', 'exag'],
                    'complexity': ['num_chars', 'num_words'],
                    'uncertainty': ['modal', 'numbers', 'question'],
                    'subjectivity': ['personal', 'quatation', 'negations', 'oov'],
                    'lexica': ['pop', 'manip', 'scandal'],
<<<<<<< HEAD
                    'formal': ['nouns', 'adjectives', 'ner']
                    }

    ## chose the group you want to eliminate
    df_feats_disag.drop(groups_feats['complexity'], inplace=True, axis=1)
    df_final['feature_input'] = df_feats_disag.iloc[:, 5:].values.tolist()  # all feats cols back to list
=======
                    'formal': ['nouns',  'adjectives',  'ner']
    }

    ## chose the group you want to eliminate
    df_feats_disag.drop(groups_feats['title'], inplace=True, axis =1)
    df_final['feature_input'] = df_feats_disag.iloc[:, 5:].values.tolist() # all feats cols back to list
>>>>>>> 5937b8681549231a4fefa4a666f2296c5e25e942
    df_final['feature_input'] = df_final['feature_input'].astype(str)

    news_train, news_test, = train_test_split(df_final, test_size=0.2, shuffle=True)
    news_test, news_dev, = train_test_split(news_test, test_size=0.5, shuffle=True)

<<<<<<< HEAD
    news_train.to_json(path_or_buf='feats_train_norm_complexity.json', force_ascii=False, orient='records')
    news_test.to_json(path_or_buf='feats_test_norm_complexity.json', force_ascii=False, orient='records')
    news_dev.to_json(path_or_buf='feats_dev_norm_complexity.json', force_ascii=False, orient='records')


if __name__ == "__main__":
    main()
=======
    news_train.to_json(path_or_buf='feats_train_norm.json', force_ascii=False, orient='records')
    news_test.to_json(path_or_buf='feats_test_norm.json', force_ascii=False, orient='records')
    news_dev.to_json(path_or_buf='feats_dev_norm.json', force_ascii=False, orient='records')


if __name__ == "__main__":
    main()
>>>>>>> 5937b8681549231a4fefa4a666f2296c5e25e942

from sklearn_genetic import GAFeatureSelectionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import yaml
from pkg_resources import resource_filename

def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def get_feats_from_col(df):
    # list to columns
    df['feature_input'] = df['feature_input'].str.replace('[', '')
    df['feature_input'] = df['feature_input'].str.replace(']', '')

    df[['num_chars', 'num_words', 'positive', 'negative', 'nouns', 'modal', 'adjectives', 'arousal', 'fear',
                      'pop', 'manip', 'scandal', 'numbers', 'ner', 'oov', 'question', 'exag', 'quatation', 'personal',
                       'hate', 'negations', 'title_num_words', 'title_question', 'title_exclam', 'title_cap']] = df.feature_input.str.split(",", expand=True, )

    df.drop('feature_input', axis=1, inplace=True)
    return df



def main():
    # get train & test data set
    print("get data")
    config = get_config('/../config/config.yaml')
    test_path = resource_filename(__name__, config['test']['path'])
    df_test = pd.read_json(test_path)

    test_path = resource_filename(__name__, config['dev']['path'])
    df_dev = pd.read_json(test_path)

    df_test_combined = pd.concat([df_test, df_dev])[['feature_input', 'label_id']]
    df_test = get_feats_from_col(df_test_combined)

    test_path = resource_filename(__name__, config['train']['path'])
    df_train = pd.read_json(test_path)[['feature_input', 'label_id']]
    df_train = get_feats_from_col(df_train)

    df_all = pd.concat([df_train, df_test])#.iloc[:1000,:]
    X = df_all.drop('label_id', axis=1)
    y = df_all['label_id']

    #train_x = df_train.drop('label_id', axis=1)
    #train_y = df_train['label_id']
    #test_x = df_test.drop('label_id', axis=1)
    #test_y = df_test['label_id']

    #clf = MLPClassifier()
    clf = SVC(gamma='auto')
    cv = StratifiedKFold(n_splits=3, shuffle=True)

    print("find best features")
    evolved_estimator = GAFeatureSelectionCV(
    estimator=clf,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=True,
    keep_top_k=2,
    elitism=True,
    )

    evolved_estimator.fit(X,y)
    print('Features:', evolved_estimator.best_features_)

if __name__ == "__main__":
    main()
<<<<<<< HEAD
=======
from sklearn_genetic import GAFeatureSelectionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
>>>>>>> 5937b8681549231a4fefa4a666f2296c5e25e942
import pandas as pd
import yaml
from pkg_resources import resource_filename
<<<<<<< HEAD
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn_genetic import GAFeatureSelectionCV
=======
from sklearn.metrics import balanced_accuracy_score, f1_score
import pickle
from sklearn.model_selection import train_test_split
>>>>>>> 5937b8681549231a4fefa4a666f2296c5e25e942


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
        'hate', 'negations', 'title_num_words', 'title_question', 'title_exclam',
        'title_cap']] = df.feature_input.str.split(",", expand=True, )

    df.drop('feature_input', axis=1, inplace=True)  # can be dropped thus later the feats are aggregated again

    df = df.astype({'num_chars': float, 'num_words': float, 'positive': float, 'negative': float, 'nouns': float,
                    'modal': float, 'adjectives': float, 'arousal': float, 'fear': float, 'pop': float,
                    'manip': float, 'scandal': float, 'numbers': float, 'ner': float, 'oov': float,
                    'question': float, 'exag': float, 'quatation': float, 'personal': float, 'hate': float,
                    'negations': float, 'title_num_words': float, 'title_question': float, 'title_exclam': float,
                    'title_cap': float, })

<<<<<<< HEAD
=======

    df.drop('feature_input', axis=1, inplace=True) #can be dropped thus later the feats are aggregated again


    df = df.astype({'num_chars': float, 'num_words': float,'positive': float,'negative': float,'nouns': float,
                    'modal': float,'adjectives': float,'arousal': float,'fear': float,'pop': float,
                    'manip': float,'scandal': float,'numbers': float,'ner': float,'oov': float,
                    'question': float,'exag': float,'quatation': float,'personal': float,'hate': float,
                    'negations': float,'title_num_words': float,'title_question': float,'title_exclam': float,'title_cap': float,})

>>>>>>> 5937b8681549231a4fefa4a666f2296c5e25e942
    return df


def main():
    # get train & test data set
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

    ## df_feats_disag is lately modificated by droping unproductive feats
    df_feats_disag = get_feats_from_col(df_all_raw)

<<<<<<< HEAD
    # for feature selection only 2 cols
    df_combined = df_feats_disag.drop(['url', 'title', 'text', 'label'], axis=1)  # .iloc[:100, :]

    # df_combined = df_combined[['num_chars', 'num_words', 'adjectives', 'ner', 'title_num_words','label_id']] #for max 3 feats
    X_train, X_test, y_train, y_test = train_test_split(df_combined.loc[:, df_combined.columns != 'label_id'],
                                                        df_combined['label_id'], test_size=0.20, random_state=42)

    clf = MLPClassifier(early_stopping=True)
    # clf = SVC()
    cv = StratifiedKFold(n_splits=3, shuffle=True)
    # '''
    print("find best features")
    evolved_estimator = GAFeatureSelectionCV(
        estimator=clf,
        cv=3,
        scoring="f1",
        generations=10,
        n_jobs=-1,
        verbose=True,
        keep_top_k=2,
        elitism=True,
        max_features=5,
        # error_score='raise'
=======

    # for feature selection only 2 cols
    df_combined = df_feats_disag.drop(['url', 'title', 'text', 'label'], axis=1)#.iloc[:100, :]

    #df_combined = df_combined[['num_chars', 'num_words', 'adjectives', 'ner', 'title_num_words','label_id']] #for max 3 feats
    X_train, X_test, y_train, y_test = train_test_split(df_combined.loc[:, df_combined.columns != 'label_id'],
                                                        df_combined['label_id'], test_size=0.20, random_state=42)


    clf = MLPClassifier(early_stopping=True)
    #clf = SVC()
    cv = StratifiedKFold(n_splits=3, shuffle=True)
    #'''
    print("find best features")
    evolved_estimator = GAFeatureSelectionCV(
    estimator=clf,
    cv=3,
    scoring="f1",
    generations=10,
    n_jobs=-1,
    verbose=True,
    keep_top_k=2,
    elitism=True,
    max_features=5,
    #error_score='raise'
>>>>>>> 5937b8681549231a4fefa4a666f2296c5e25e942
    )

    evolved_estimator.fit(X_train, y_train)
    features = evolved_estimator.best_features_
<<<<<<< HEAD
    # pickle.dump(features, open('features.p', "wb"))
=======
    #pickle.dump(features, open('features.p', "wb"))
>>>>>>> 5937b8681549231a4fefa4a666f2296c5e25e942
    print(features)
    prod_feats_idx = [i for i, elem in enumerate(features.tolist()) if elem is True]
    prod_feats = [X_test.columns[i] for i in prod_feats_idx]
    print(prod_feats)

    y_predict_ga = evolved_estimator.predict(X_test[prod_feats])
    print('the accuracy is:', f1_score(y_test, y_predict_ga))
<<<<<<< HEAD
    # '''

    # feats = pickle.load(open("features.p", "rb"))
=======
    #'''


    #feats = pickle.load(open("features.p", "rb"))
>>>>>>> 5937b8681549231a4fefa4a666f2296c5e25e942

    '''
    # aggregate chosen feats
    df_final = df_feats_disag[['url','title','text','label','label_id']]

    df_final['feature_input'] = df_feats_disag[prod_feats].values.tolist()
    df_final['feature_input'] = df_final['feature_input'].astype(str)

    # split & save
    news_train, news_test, = train_test_split(df_final, test_size=0.2, shuffle=True)
    news_test, news_dev, = train_test_split(news_test, test_size=0.5, shuffle=True)

    news_train.to_json(path_or_buf='feats_train_norm.json', force_ascii=False, orient='records')
    news_test.to_json(path_or_buf='feats_test_norm.json', force_ascii=False, orient='records')
    news_dev.to_json(path_or_buf='feats_dev_norm.json', force_ascii=False, orient='records')
    '''
<<<<<<< HEAD


if __name__ == "__main__":
    main()
=======

if __name__ == "__main__":
    main()


>>>>>>> 5937b8681549231a4fefa4a666f2296c5e25e942

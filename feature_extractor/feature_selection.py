from sklearn_genetic import GAFeatureSelectionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import yaml
from pkg_resources import resource_filename
from sklearn.metrics import balanced_accuracy_score, f1_score
import pickle
from sklearn.model_selection import train_test_split
from scipy import stats
import shap
import matplotlib.pyplot as pl


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def get_feats_from_col(df):
    # list to columns
    df['feature_input'] = df['feature_input'].str.replace('[', '')
    df['feature_input'] = df['feature_input'].str.replace(']', '')

    df[['num_chars', 'num_words', 'positive', 'negative', 'nouns', 'modal', 'adjectives', 'arousal',
                      'style', 'anto', 'topoi', 'numbers', 'ner', 'oov', 'question',  'quatation', 'exclamation','personal',
                       'hate', 'negations', 'title_num_words', 'title_question', 'title_exclam', 'title_cap']] = df.feature_input.str.split(",", expand=True, )


    df.drop('feature_input', axis=1, inplace=True) #can be dropped thus later the feats are aggregated again


    df = df.astype({'num_chars': float, 'num_words': float,'positive': float,'negative': float,'nouns': float,
                    'modal': float,'adjectives': float,'arousal': float,'style': float,
                    'anto': float,'topoi': float,'numbers': float,'ner': float,'oov': float,
                    'question': float,'quatation': float, 'exclamation': float,'personal': float,'hate': float,
                    'negations': float,'title_num_words': float,'title_question': float,'title_exclam': float,'title_cap': float,})

    return df


def select_features(df_combined):

    X_train, X_test, y_train, y_test = train_test_split(df_combined.loc[:, df_combined.columns != 'label_id'],
                                                        df_combined['label_id'], test_size=0.20, random_state=42)

    #clf = MLPClassifier(early_stopping=True)
    clf = LogisticRegression()
    cv = StratifiedKFold(n_splits=3, shuffle=True)

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
        #max_features=5,
        # error_score='raise'
    )

    evolved_estimator.fit(X_train, y_train)
    features = evolved_estimator.best_features_
    # pickle.dump(features, open('features.p', "wb"))
    print(features)
    prod_feats_idx = [i for i, elem in enumerate(features.tolist()) if elem is True]
    prod_feats = [X_test.columns[i] for i in prod_feats_idx]
    print(prod_feats)

    y_predict_ga = evolved_estimator.predict(X_test[prod_feats])
    print('the accuracy is:', f1_score(y_test, y_predict_ga))
    return prod_feats

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

    # drop feats not passing t-test
    df_true = df_feats_disag[df_feats_disag['label_id'] == 1]
    df_false = df_feats_disag[df_feats_disag['label_id'] == 0]
    for i in df_feats_disag.columns.tolist()[5:]:
        if stats.ttest_ind(df_true[i], df_false[i])[1] > 0.01:
            print(i)
            df_feats_disag.drop(columns=i, inplace = True)


    # select feats
    df_feats_disag.drop(['url', 'title', 'text', 'label'], axis=1, inplace=True)
    #df_combined = df_feats_disag[['modal', 'arousal', 'nouns' ,'title_question', 'label_id']] #for max 3 feats
    #chosen_feats = select_features(df_feats_disag)
    clf = MLPClassifier(early_stopping=True, learning_rate_init=0.00001, batch_size=2,
                        hidden_layer_sizes=(2, 128, 64, 2),
                        activation='logistic')
    X_train, X_test, y_train, y_test = train_test_split(df_feats_disag.loc[:, df_feats_disag.columns != 'label_id'],
                                                          df_feats_disag['label_id'], test_size=0.20, random_state=42)
    clf.fit(X_train.values, y_train.values)

    explainer = shap.KernelExplainer(clf.predict, X_train.values)
    shap_values = explainer.shap_values(X_test.values, nsamples=100)
    shap.summary_plot(shap_values, X_test,
                      feature_names=['num_chars', 'num_words', 'positive', 'negative', 'nouns', 'modal', 'adjectives', 'arousal',
                      'style', 'anto', 'topoi', 'numbers', 'ner', 'oov', 'question',  'quatation', 'exclamation','personal',
                       'hate', 'negations', 'title_num_words', 'title_question', 'title_exclam', 'title_cap'],
                      show=False)

    pl.savefig("shap_summary.svg", dpi=700)

    '''
    chosen_feats = ['num_chars', 'num_words', 'adjectives', 'fear', 'pop', 'manip', 'scandal', 'numbers', 'ner', 'oov', 'question', 'exag', 'quatation', 'personal', 'hate', 'negations', 'title_num_words', 'title_exclam', 'title_cap']

    # get new train, test, dev with chosen features
    # aggregate chosen feats
    df_final = df_feats_disag[['url', 'title', 'text', 'label', 'label_id']]
    df_final['feature_input'] = df_feats_disag[chosen_feats].values.tolist()
    df_final['feature_input'] = df_final['feature_input'].astype(str)

    # split & save
    news_train, news_test, = train_test_split(df_final, test_size=0.2, shuffle=True)
    news_test, news_dev, = train_test_split(news_test, test_size=0.5, shuffle=True)

    news_train.to_json(path_or_buf='feats_train_norm.json', force_ascii=False, orient='records')
    news_test.to_json(path_or_buf='feats_test_norm.json', force_ascii=False, orient='records')
    news_dev.to_json(path_or_buf='feats_dev_norm.json', force_ascii=False, orient='records')
    '''
if __name__ == "__main__":
    main()



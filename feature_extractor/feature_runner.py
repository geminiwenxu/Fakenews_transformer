import json
import pandas as pd
import yaml
from pkg_resources import resource_filename
from transformers import AutoModel, pipeline
from draft_feats import Article, Headline
import pickle
import spacy
from sklearn import preprocessing




def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def main():
    config = get_config('/../config/config.yaml')
    train_path = resource_filename(__name__, config['train']['path'])
    df = pd.read_json(train_path)

    nlp = spacy.load('de_core_news_lg')
    # get dictionaries for positive, negative and arousal words
    # load keyword list
    dict_positive = pickle.load(open("dictPositive.p", "rb"))
    dict_negative = pickle.load(open("dictNegative.p", "rb"))
    #keywords = pd.read_csv('Keywords_fake.txt')
    #keywords = keywords.keyword.str.lower().tolist()
    pd_arousal = pd.read_csv(filepath_or_buffer='list_arousal.csv', sep=';')
    dict_arousal = pd_arousal.set_index('WORD_LOWER').to_dict()['AROUSAL_MEAN']


    # keywords list
    # fear
    fear = pd.read_csv('keywords_fear.txt')
    fear = fear.keyword.str.lower().tolist()
    #scandal
    scandal = pd.read_csv('keywords_scandal.txt')
    scandal = scandal.keyword.str.lower().tolist()
    #populism
    polulism = pd.read_csv('keywords_populism.txt')
    polulism = polulism.keyword.str.lower().tolist()

    # manipulation
    manipulation = pd.read_csv('keywords_manipulation.txt')
    manipulation = manipulation.keyword.str.lower().tolist()

    # for hatespeech detection
    model_name = 'EIStakovskii/german_toxicity_classifier_plus_v2'
    toxicity_pipeline = pipeline('text-classification', model=model_name, tokenizer=model_name)


    df_new = df.iloc[:,:]

    list_feats = []
    for index, row in df_new.iterrows():
        feats_article = Article(raw_text=row['text'],
                                dict_positive=dict_positive, dict_negative=dict_negative, keywords_fear=fear, keywords_scandal=scandal, keywords_pop=polulism,keywords_manip=manipulation,
                                dict_arousal=dict_arousal, toxicity_pipeline=toxicity_pipeline, nlp=nlp
                                )
        #print(feats_article.return_results())

        feats_headline = Headline(row['title'], nlp)
        #print(feats_headline.return_results())

        feats_all = feats_article.return_results() + feats_headline.return_results()

        list_feats.append(feats_all)

    df_feats = pd.DataFrame(list_feats)


    # replace outliers with quantiles
    '''
    for i in range(0, 24):
            lower_limit = df_feats[i].quantile(0.05)
            upper_limit = df_feats[i].quantile(0.95)
            df_feats[i] = np.where(df_feats[i] > upper_limit, upper_limit,
                                   np.where(df_feats[i] < lower_limit, lower_limit,
                                            df_feats[i]))
    '''
    scaler = preprocessing.RobustScaler(quantile_range=(5.0, 95.0))
    df_feats_scaled = scaler.fit_transform(df_feats)
    df_feats_scaled = pd.DataFrame(df_feats_scaled)

    # save the feats for furher inspection
    df_feats['label_id'] = df_new['label_id']
    df_feats['url'] = df_new['url']  #act as a key for later join
    df_feats.to_csv('feats_train2.csv')

    df_new['feature_input'] = df_feats_scaled.values.tolist()
    df_new['feature_input'] = df_new['feature_input'].astype(str)

    if df_new[df_new['feature_input'].str.contains('nan')].size != 0:
        print("nan values removed")
        df_new = df_new.drop([df_new.index[df_new[df_new['feature_input'].str.contains('nan')].index[0]]])

    df_new.to_json(path_or_buf='news_feats_train_norm.json', force_ascii=False, orient='records')


if __name__ == "__main__":
    main()

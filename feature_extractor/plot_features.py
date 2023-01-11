import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from data_scraping import get_data_statistics as d
import plotly.express as px
import yaml
from pkg_resources import resource_filename
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf

"""
lst_names = ['num_chars', 'num_words', 'positive', 'negative', 'nouns', 'modal', 'adjectives', 'arousal',
            'style', 'anto', 'topoi', 'numbers', 'ner', 'oov', 'question', 'quatation', 'exclamation', 'personal',
            'hate', 'negations', 'title_num_words', 'title_question', 'title_exclam', 'title_cap',
            'label_id', 'url']

lst_feats = lst_names[0:-2]

df_feats = pd.read_csv('feats_raw_train.csv', header=0,names = lst_names)

try:
    df_feats.drop(df_feats[df_feats['url'].duplicated()].index[0], inplace=True)
except:
    pass

# needed later to join again the feats
df_all_raw_text =  pd.read_json(resource_filename(__name__, '/../data_scraping/data_all.json'))

try:
    df_all_raw_text.drop(df_all_raw_text[df_all_raw_text['url'].duplicated()].index[0], inplace=True)
except:
    pass


df_combined = df_all_raw_text.merge(df_feats, left_on = 'url', right_on = 'url')
df_combined.drop('label_id_y', inplace=True, axis=1)
df_combined.rename(columns={'label_id_x':'label_id'}, inplace=True)

pickle.dump(df_combined, open( "save.p", "wb" ) )
"""
df_combined = pickle.load( open( "save.p", "rb" ) )

## plotting ###

df_true = df_combined[df_combined['label_id'] == 1].iloc[:10000,]
df_false = df_combined[df_combined['label_id'] == 0].iloc[:10000,]

df_plot_true = df_true[['adjectives','nouns']].melt()
df_plot_true['label'] = 'True'
df_plot_false = df_false[['adjectives','nouns']].melt()
df_plot_false['Label'] = 'Fake'
df_plt = pd.concat([df_plot_true, df_plot_false])

palette = {
    'Fake': '#CD5C5C',
    'True': '#8FBC8F',
}

plt.figure(figsize=(10,7))
sns.catplot(data=df_plt, kind="box", x="variable", y="value", hue="Label", palette=palette)
plt.title("test")
plt.show()

"""
df_true = df_combined[df_combined['label_id'] == 1]
df_true = df_true[lst_feats].mean(axis=0)
df_false = df_combined[df_combined['label_id'] == 0]
df_false = df_false[lst_feats].mean(axis=0)

df = pd.DataFrame({'true': df_true,
                   'false': df_false, 'feature': lst_feats})

fig = px.bar(df, x=['true', 'false'], y='feature', orientation='h', barmode="group", #width=800, height=600,
                    labels = {
                        "value": "Features",
                        "variable": "Label"
                    },
                    color_discrete_sequence=['green', 'red']
)
fig.show()

"""
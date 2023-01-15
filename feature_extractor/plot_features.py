import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from pkg_resources import resource_filename


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def plot_selected_features(lst_features):
    df_plot_true = df_true[lst_features].melt()
    df_plot_true['Label'] = 'True'
    df_plot_false = df_false[lst_features].melt()
    df_plot_false['Label'] = 'Fake'
    df_plt = pd.concat([df_plot_true, df_plot_false])

    palette = {
        'Fake': '#CD5C5C',
        'True': '#8FBC8F',
    }

    plt.figure(figsize=(10, 7))
    ax = sns.boxplot(data=df_plt, x="variable", y="value", hue="Label", palette=palette)
    ax.set(xlabel=None)

    plt.legend([], [], frameon=False)
    name_plt = lst_features[0]
    plt.savefig(name_plt)


lst_names = ['num_chars', 'num_words', 'positive', 'negative', 'nouns', 'modal', 'adjectives', 'arousal',
             'style', 'anto', 'topoi', 'numbers', 'ner', 'oov', 'question', 'quatation', 'exclamation', 'personal',
             'hate', 'negations', 'title_num_words', 'title_question', 'title_exclam', 'title_cap',
             'label_id', 'url']

lst_feats = lst_names[0:-2]

df_feats = pd.read_csv('feats_raw_all.csv', header=0, names=lst_names)

# needed later to join again the feats
df_all_raw_text = pd.read_json(resource_filename(__name__, '/../data_scraping/data_all.json'))

try:
    df_all_raw_text.drop(df_all_raw_text[df_all_raw_text['url'].duplicated()].index[0], inplace=True)
except:
    pass

df_combined = df_all_raw_text.merge(df_feats, left_on='url', right_on='url')
df_combined.drop('label_id_y', inplace=True, axis=1)
df_combined.rename(columns={'label_id_x': 'label_id'}, inplace=True)

# get rid of outliers 0.05/0.95

for col in df_combined.columns[5:]:
    print("capping the ", col)
    if (((df_combined[col].dtype) == 'float64') | ((df_combined[col].dtype) == 'int64')):
        percentiles = df_combined[col].quantile([0.05, 0.95]).values
        df_combined[col][df_combined[col] <= percentiles[0]] = percentiles[0]
        df_combined[col][df_combined[col] >= percentiles[1]] = percentiles[1]
    else:
        df_combined[col] = df_combined[col]

## plotting ###
df_true = df_combined[df_combined['label_id'] == 1].iloc[:, ]
df_false = df_combined[df_combined['label_id'] == 0].iloc[:, ]

# plot features together if same category and same value range
features_plot_together = [['num_chars'],
                          ['num_words'],
                          ['negative', 'positive'],
                          ['nouns', 'modal', 'adjectives'],
                          ['numbers', 'ner', 'oov'],
                          ['question', 'quatation', 'exclamation'],
                          ['hate', 'arousal'],
                          ['title_num_words', 'title_cap'],
                          ['title_exclam', 'title_question'],
                          ['anto', 'topoi', 'style', 'personal']
                          ]

for i in features_plot_together:
    plot_selected_features(i)

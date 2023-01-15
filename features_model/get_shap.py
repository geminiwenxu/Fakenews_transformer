import ast

import matplotlib.pyplot as pl
import pandas as pd
import shap
import torch
import yaml
from pkg_resources import resource_filename

from features_model.models import FakeNewsBinaryModel


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/config.yaml')

train_path = resource_filename(__name__, config['train']['path'])
dev_path = resource_filename(__name__, config['dev']['path'])
test_path = resource_filename(__name__, config['test']['path'])
BATCH_SIZE = config['batch_size']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df_train = pd.read_json(train_path)
df_dev = pd.read_json(dev_path)
df_test = pd.read_json(test_path)

model = FakeNewsBinaryModel(batch_size=BATCH_SIZE)
model.to(device)

model.load_state_dict(torch.load('best_model_state.bin', map_location='cpu'))

feature_input = df_train["feature_input"]
y = []
for i in feature_input:
    x = ast.literal_eval(i)
    y.append(x)
feature_input = torch.tensor(y).to(torch.float32).to(device)

features = ['num_chars', 'num_words', 'positive', 'negative', 'nouns', 'modal', 'adjectives', 'arousal',
            'style', 'anto', 'topoi', 'numbers', 'ner', 'oov', 'question', 'quatation', 'exclamation', 'personal',
            'hate', 'negations', 'title_num_words', 'title_question', 'title_exclam', 'title_cap']
torch.set_grad_enabled(True)

e = shap.DeepExplainer(
    model, feature_input[torch.randperm(len(feature_input))[:10000]])

x_samples = feature_input[torch.randperm(len(feature_input))[:1000]]

shap_values = e.shap_values(x_samples.to(device))

shap.summary_plot(shap_values, features=x_samples, feature_names=features, show=False)
pl.savefig("shap_summary_all.png", dpi=700)

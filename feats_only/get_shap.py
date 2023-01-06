from collections import defaultdict
import ast
import pandas as pd
import torch
import yaml
from pkg_resources import resource_filename
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
import numpy as np
from feats_only.models import FakeNewsBinaryModel
from prediction import get_predictions
from feats_only.prepare_data import create_data_loader
from train import train_epoch, eval_model
from utilities.log_samples import save_samples
from utilities.plot import plot
from feature_extractor.feature_selection import get_feats_from_col
import shap
import matplotlib.pyplot as pl


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

# x_samples = feature_input.sample(n=10, replace=True, random_state=1).to_numpy()

x_samples = feature_input[torch.randperm(len(feature_input))[:1000]]
print(len(x_samples))
shap_values = e.shap_values(x_samples.to(device))

df = pd.DataFrame({
    "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
    "stdev_abs_shap": np.std(np.abs(shap_values), axis=0),
    "name": features
})
df.sort_values("mean_abs_shap", ascending=False)#[:10]

df.to_csv("feature_importances.csv", index=False)

shap.summary_plot(shap_values, features=x_samples, feature_names=features, show=False, max_display=24)
pl.savefig("shap_summary.png", dpi=700)
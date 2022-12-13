import pandas as pd
import torch
import yaml
from pkg_resources import resource_filename


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def weights():
    config = get_config('/../../config/config.yaml')
    train_path = resource_filename(__name__, config['train']['path'])
    dev_path = resource_filename(__name__, config['dev']['path'])
    test_path = resource_filename(__name__, config['test']['path'])
    num_real =0
    num_fake =0
    for path in [train_path, dev_path, test_path]:
        df = pd.read_json(path)
        num_real += len(df[df['label_id'] == 1])
        num_fake += len(df[df['label_id'] == 0])
    w_0 = (num_fake + num_real) / (2.0 * num_fake)
    w_1 = (num_fake + num_real) / (2.0 * num_real)
    class_weights = torch.FloatTensor([w_0, w_1])
    return class_weights

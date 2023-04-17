import json

import yaml
from pkg_resources import resource_filename


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


if __name__ == '__main__':
    config = get_config('/../config/config.yaml')
    train_path = resource_filename(__name__, config['train']['path'])
    dev_path = resource_filename(__name__, config['dev']['path'])
    test_path = resource_filename(__name__, config['test']['path'])

    with open(train_path, 'r') as file:
        json_data = json.load(file)
        for i in json_data:
            label = int(i['label_id'])
            i['label_id'] = label
    with open('int_train.json', 'w') as file:
        json.dump(json_data, file, indent=2)

    with open(dev_path, 'r') as file:
        json_data = json.load(file)
        for i in json_data:
            label = int(i['label_id'])
            i['label_id'] = label
    with open('int_dev.json', 'w') as file:
        json.dump(json_data, file, indent=2)

    with open(test_path, 'r') as file:
        json_data = json.load(file)
        for i in json_data:
            label = int(i['label_id'])
            i['label_id'] = label
    with open('int_test.json', 'w') as file:
        json.dump(json_data, file, indent=2)

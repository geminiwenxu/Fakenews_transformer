from collections import defaultdict

import pandas as pd
import torch
import torch.nn as nn
import yaml
from pkg_resources import resource_filename
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from fusion_model.models import FakeNewsBinaryModel
from baseline_model.plot import plot
from fusion_model.prediction import get_predictions
from fusion_model.prepare_data import create_data_loader
from fusion_model.train import train_epoch, eval_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ['real', 'fake']
tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased', do_lower_case=True)

MAX_LEN = 160
BATCH_SIZE = 2


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


config = get_config('/../config/config.yaml')
train_path = resource_filename(__name__, config['train']['path'])
dev_path = resource_filename(__name__, config['dev']['path'])
test_path = resource_filename(__name__, config['test']['path'])

df_train = pd.read_json(train_path)
df_dev = pd.read_json(dev_path)
df_test = pd.read_json(test_path)

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
dev_data_loader = create_data_loader(df_dev, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

model = FakeNewsBinaryModel(batch_size=BATCH_SIZE)
model.to(device)
EPOCHS = 5
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
loss_fn = nn.BCELoss().to(device)


def main():
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            dev_data_loader,
            loss_fn,
            device,
            len(df_dev)
        )

        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(val_acc)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc
    plot(history)
    y_review_texts, y_pred, y_pred_probs, y_dev = get_predictions(
        model,
        test_data_loader
    )
    y_pred = y_pred.cpu().detach().numpy()

    print("pred y_pred", y_pred)
    print("pred y_dev", y_dev)
    print(classification_report(y_dev, y_pred, target_names=class_names))


if __name__ == '__main__':
    main()

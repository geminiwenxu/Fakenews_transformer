import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel, BertModel, AutoTokenizer
if __name__ == '__main__':
    model_name_or_path = 'deepset/gbert-base'
    model: BertModel = AutoModel.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    chosen_feature_size = 128
    # cls_encoder = nn.Linear(last_n_layers * model.config.hidden_size, chosen_feature_size)
    texts = [f"Ein normaler Zeitungsartikel {i}..." for i in range(3)] + [f"Eine Verschwörungstheorie {i}..." for i in range(2)]
    tokenized = tokenizer(texts, return_tensors='pt', padding='max_length')
    batch_size = len(texts)
    n_textual_features = 25
    # Some random numbers inplace of your features
    features = torch.rand((batch_size, n_textual_features))

    labels = torch.tensor([0,0,0,1,1]).to(torch.long)

    # Example number of samples per label:
    samples_per_class = torch.tensor([1000, 400])
    class_weights = samples_per_class.max().div(samples_per_class)  # -> [1.0, 2.5]
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # print(tokenized)
    # print(features)
    # print(labels)
    print(samples_per_class)
    print(class_weights)
    print(samples_per_class.max())
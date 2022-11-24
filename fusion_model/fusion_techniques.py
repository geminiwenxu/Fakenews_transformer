import torch


def concatenate(bert_embedding, feature_embedding):
    return torch.cat((bert_embedding, feature_embedding), dim=0)


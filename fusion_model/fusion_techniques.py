import torch


def concatenate(bert_embedding, feature_embedding):
    return torch.cat((bert_embedding, feature_embedding), dim=0)


def attention_fusion(bert_embedding, feature_embedding):
    pass
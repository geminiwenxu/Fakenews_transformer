import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def concatenate(bert_embedding, feature_embedding):
    return torch.cat((bert_embedding, feature_embedding), dim=0)


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(k.reshape(-1, 1), q.reshape(1, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


def attention_fusion(bert_embedding, feature_embedding):
    head = 1
    attention_layer = 1

    for layer in range(attention_layer):
        for _ in range(head):
            linear1 = nn.Linear(512, 32, bias=True).to(device)
            t_f_K = linear1(feature_embedding)
            print(linear1.training)
            linear2 = nn.Linear(768, 32, bias=True).to(device)
            t_f_Q = linear2(bert_embedding)
            linear3 = nn.Linear(512, 32, bias=True).to(device)
            t_f_V = linear3(feature_embedding)
            t_f_value, t_f_attention = scaled_dot_product(t_f_Q, t_f_K, t_f_V)

            linear4 = nn.Linear(512, 32, bias=True).to(device)
            f_t_Q = linear4(feature_embedding)

            linear5 = nn.Linear(768, 32, bias=True).to(device)
            f_t_K = linear5(bert_embedding)

            linear6 = nn.Linear(768, 32, bias=True).to(device)
            f_t_V = linear5(bert_embedding)

            f_t_value, f_t_attention = scaled_dot_product(f_t_Q, f_t_K, f_t_V)
    value, inx = torch.max(torch.stack((f_t_value, t_f_value), dim=0), dim=0)
    return torch.cat((f_t_value, t_f_value))

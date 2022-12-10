import math

import torch
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
            feature_key = torch.randn(512, 32).to(device)
            text_query = torch.randn(768, 32).to(device)
            feature_value = torch.randn(512, 32).to(device)
            t_f_Q = torch.mv(torch.t(text_query), bert_embedding)
            t_f_K = torch.mv(torch.t(feature_key), feature_embedding)
            t_f_V = torch.mv(torch.t(feature_value), feature_embedding)
            t_f_value, t_f_attention = scaled_dot_product(t_f_Q, t_f_K, t_f_V)

            text_key = torch.randn(768, 32).to(device)
            feature_query = torch.randn(512, 32).to(device)
            text_value = torch.randn(768, 32).to(device)
            f_t_Q = torch.mv(torch.t(feature_query), feature_embedding)
            f_t_K = torch.mv(torch.t(text_key), bert_embedding)
            f_t_V = torch.mv(torch.t(text_value), bert_embedding)
            f_t_value, f_t_attention = scaled_dot_product(f_t_Q, f_t_K, f_t_V)
    value, inx = torch.max(torch.stack((f_t_value, t_f_value), dim=0), dim=0)
    return value

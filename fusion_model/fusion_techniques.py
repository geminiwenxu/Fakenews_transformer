import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FinalNet(nn.Module):
    def __int__(self):
        super(FinalNet, self).__init__()
        self.layer1 = nn.Linear(32, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 32)
        self.layer4 = nn.Linear(32, 32)
        self.final_layer = nn.Linear(32, 32)
        self.norm1 = torch.nn.LayerNorm(32)

    def forward(self, att_layer, query):
        output_layer1 = self.layer1(att_layer)
        output_layer2 = self.layer2(att_layer)
        output_layer3 = self.layer3(att_layer)
        output_layer4 = self.layer4(att_layer)
        max_layer, inx = torch.max(torch.stack((output_layer1, output_layer2, output_layer3, output_layer4), dim=0),
                                   dim=0)
        final_output_layer = self.final_layer(max_layer)

        # add & norm
        add_layer = final_output_layer + query
        return self.norm1(add_layer)


def concatenate(bert_embedding, feature_embedding):
    return torch.cat((bert_embedding, feature_embedding), dim=0)


def get_final_output(input_att, query):
    """
    this function implements the processing of final outputs from attention infusion as described in Tuan&Minh(2021)
    every vector is passed thought 4 fully connected layers and the max is chosen than added to query (add conncection)
    lastly batch layer normalization is proceeded
    :param input_att: output of attention infusion
    :param query: query for add
    :return:final vector
    """
    layer1 = nn.Linear(32, 32).to(device)
    layer2 = nn.Linear(32, 32).to(device)
    layer3 = nn.Linear(32, 32).to(device)
    layer4 = nn.Linear(32, 32).to(device)
    final_layer = nn.Linear(32, 32).to(device)
    norm1 = torch.nn.LayerNorm(32).to(device)

    output_layer1 = layer1(input_att)
    output_layer2 = layer2(input_att)
    output_layer3 = layer3(input_att)
    output_layer4 = layer4(input_att)
    max_layer, inx = torch.max(torch.stack((output_layer1, output_layer2, output_layer3, output_layer4), dim=0),
                               dim=0)
    final_output_layer = final_layer(max_layer)

    # add & norm
    add_layer = final_output_layer + query
    final_output = norm1(add_layer).to(device)
    return final_output


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
            f_t_V = linear6(bert_embedding)

            f_t_value, f_t_attention = scaled_dot_product(f_t_Q, f_t_K, f_t_V)

            # fully connected layers as in paper

            # model = FinalNet()
            # output_t = model(t_f_value, t_f_Q)
            # output_f = model(f_t_value, f_t_Q)
            # ToDo why this is working in function but not in class? Hopefully not wrong
            output_t = get_final_output(t_f_value, t_f_Q)
            output_f = get_final_output(f_t_value, f_t_Q)

            # the attention blocks are concat with the inputs for text and features
            output_cat = torch.cat((bert_embedding, feature_embedding, output_t, output_f), dim=0)

    # value, inx = torch.max(torch.stack((f_t_value, t_f_value), dim=0), dim=0)
    return output_cat

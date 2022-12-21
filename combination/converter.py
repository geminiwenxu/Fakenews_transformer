import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import AutoModel
from transformers import AutoTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BertConverter():
    def __init__(self):
        super(BertConverter, self).__init__()
        self.model = AutoModel.from_pretrained('bert-base-german-cased', output_hidden_states=True).to(device)
        self.activation = nn.GELU()
        self.cls_encoder = nn.Linear(4 * self.model.config.hidden_size, 128).to(device)
        self.model.train()

    def forward(self, input_ids, attention_mask):
        bert_results = self.model(input_ids, attention_mask)
        encoded_layers = torch.cat(bert_results['hidden_states'][-4:], -1)[:, 0]
        encoded_cls = self.cls_encoder(encoded_layers)
        return self.activation(encoded_cls)  # encoded_cls


class TextConverter(BertConverter):
    def __init__(self, batch_size):
        super(TextConverter, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased', do_lower_case=False)
        self.text_encoder = nn.LSTM(self.model.config.hidden_size, 128, bidirectional=True).to(device)
        self.batch_size = batch_size

    def encode(self, input_ids, attention_mask, length):
        bert_results = self.model(input_ids, attention_mask)
        v_length = torch.flatten(length)
        packed = pack_padded_sequence(bert_results.last_hidden_state, v_length.cpu(), batch_first=True, enforce_sorted=False)
        _, (rnn_last_hidden_state, _) = self.text_encoder(packed)
        encoded_text = rnn_last_hidden_state.view(self.batch_size, -1)
        return encoded_text


class FeatureConverter(nn.Module):
    def __init__(self, batch_size):
        super(FeatureConverter, self).__init__()
        self.batch_size = batch_size
        self.layer1 = nn.Linear(25, self.batch_size)
        self.layer2 = nn.Linear(self.batch_size, 128)
        self.activation = nn.GELU()

    def forward(self, feature_input):
        hidden_output_1 = self.layer1(feature_input)
        hidden_output_2 = self.layer2(hidden_output_1)
        return self.activation(hidden_output_2)  # 128 encoded_feature


class DenseConverter(nn.Module):
    def __init__(self, batch_size):
        super(DenseConverter, self).__init__()
        self.batch_size = batch_size
        self.classifier = nn.Linear(512, 1)
        self.drop = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoded_cls, encoded_text, encoded_features):
        joint_embedding = torch.cat((encoded_cls, encoded_text, encoded_features), -1)
        hidden_output = self.classifier(joint_embedding)
        drop_output = self.drop(hidden_output)
        prob = self.sigmoid(drop_output)
        return torch.flatten(prob)

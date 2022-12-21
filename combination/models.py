import torch
import torch.nn as nn

from converter import BertConverter, TextConverter, FeatureConverter, DenseConverter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FakeNewsBinaryModel(nn.Module):
    def __init__(self, batch_size):
        super(FakeNewsBinaryModel, self).__init__()
        self.batch_size = batch_size
        self.bert_converter = BertConverter()
        self.text_converter = TextConverter(batch_size=self.batch_size)
        self.feature_converter = FeatureConverter(batch_size=self.batch_size)
        self.dense_converter = DenseConverter(batch_size=self.batch_size)

    def forward(self, input_ids, attention_mask, length, **feature_input):
        bert_embedding = torch.tensor(self.bert_converter.forward(input_ids, attention_mask))
        feature_embedding = self.feature_converter.forward(feature_input['feature_inputs'])
        text_embedding = self.text_converter.encode(input_ids, attention_mask, length)
        prob = self.dense_converter(bert_embedding, text_embedding, feature_embedding)
        return prob

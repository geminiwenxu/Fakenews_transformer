import torch
import torch.nn as nn

from converter import BertConverter, FeatureConverter, DenseConverter
from fusion_techniques import concatenate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FakeNewsBinaryModel(nn.Module):
    def __init__(self, batch_size):
        super(FakeNewsBinaryModel, self).__init__()
        self.batch_size = batch_size
        self.bert_converter = BertConverter()
        self.feature_converter = FeatureConverter(batch_size=self.batch_size)
        self.dense_converter = DenseConverter(batch_size=self.batch_size)

    def forward(self, input_ids, attention_mask, **feature_input):
        bert_embedding = torch.FloatTensor(self.bert_converter.forward(input_ids, attention_mask)).to(device)
        feature_embedding = self.feature_converter.forward(feature_input['feature_inputs'])
        joint_embedding = concatenate(bert_embedding, feature_embedding)
        prob = self.dense_converter(joint_embedding)
        return prob
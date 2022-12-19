import torch
import torch.nn as nn

from converter import BertConverter, FeatureConverter, AttenDenseConverter
from fusion_techniques import attention_fusion

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FakeNewsBinaryModel(nn.Module):
    def __init__(self, batch_size):
        super(FakeNewsBinaryModel, self).__init__()
        self.batch_size = batch_size
        self.bert_converter = BertConverter()
        self.feature_converter = FeatureConverter(batch_size=self.batch_size)
        self.atten_dense_converter = AttenDenseConverter(batch_size=self.batch_size)

    def forward(self, input_ids, attention_mask, **feature_input):
        bert_embedding = torch.tensor(self.bert_converter.forward(input_ids, attention_mask))  # [16, 160, 3072]
        bert_embedding = torch.flatten(bert_embedding, start_dim=0, end_dim=-1)  # [7864320]
        feature_embedding = self.feature_converter.forward(feature_input['feature_inputs'])
        joint_embedding = attention_fusion(bert_embedding, feature_embedding)
        prob = self.atten_dense_converter(joint_embedding)
        return prob

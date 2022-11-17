import torch

from converter import BertConverter, FeatureConverter, DenseConverter


class FakeNewsBinaryModel(BertConverter, FeatureConverter, DenseConverter):
    def __init__(self, output_dim):
        super(FakeNewsBinaryModel, self).__init__()
        self.bert_converter = BertConverter()
        self.feature_converter = FeatureConverter()
        self.dense_converter = DenseConverter(output_dim)

    def forward(self, input_ids, attention_mask, feature_input=torch.tensor([1, 2, 3]).to(torch.float32)):
        bert_embedding = torch.FloatTensor(self.bert_converter.forward(input_ids, attention_mask))
        feature_embedding = self.feature_converter.forward(feature_input)
        joint_embedding = torch.cat((bert_embedding, feature_embedding), dim=0)
        prob = self.dense_converter(joint_embedding)
        print('outputs of model', prob)
        return prob

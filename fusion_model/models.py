import torch

from converter import BertConverter, FeatureConverter, DenseConverter


class FakeNewsBinaryModel(BertConverter, FeatureConverter, DenseConverter):
    def __init__(self):
        super(FakeNewsBinaryModel, self).__init__()
        self.bert_converter = BertConverter()
        self.feature_converter = FeatureConverter()
        self.dense_converter = DenseConverter()

    def classifier(self, sentences, feature_input):
        bert_embedding = self.bert_converter.encode_to_vec(sentences)
        feature_embedding = self.feature_converter.forward(feature_input)
        joint_embedding = torch.cat((bert_embedding, feature_embedding), dim=1)
        prob = self.dense_converter(joint_embedding)
        return prob

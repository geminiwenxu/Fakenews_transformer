import torch

from converter import BertConverter, FeatureConverter, DenseConverter


class FakeNewsBinaryModel(BertConverter, FeatureConverter, DenseConverter):
    def __init__(self):
        super(FakeNewsBinaryModel, self).__init__()
        self.bert_converter = BertConverter()
        self.feature_converter = FeatureConverter()
        self.dense_converter = DenseConverter()

    def forward(self, input_ids, attention_mask, **feature_input):
        bert_embedding = torch.FloatTensor(self.bert_converter.forward(input_ids, attention_mask))
        print("bert embeddings",bert_embedding, bert_embedding.size())
        print(feature_input)
        feature_embedding = self.feature_converter.forward(feature_input['feature_inputs'])
        print(feature_embedding, feature_embedding.size())
        joint_embedding = torch.cat((bert_embedding, feature_embedding), dim=0)
        print('joint', joint_embedding.size())
        prob = self.dense_converter(joint_embedding)
        print('outputs of model', prob)
        return prob

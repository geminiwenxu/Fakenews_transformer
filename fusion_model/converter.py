import torch.nn as nn

from transformers import AutoModel


class BertConverter:
    def __init__(self):
        self.model = AutoModel.from_pretrained('bert-base-german-cased')
        self.model.eval()  # put the model in eval mode, meaning feed-forward operation?

    def encode_to_vec(self, input_ids, attention_mask):
        bert_results = self.model(input_ids, attention_mask)
        return bert_results.last_hidden_state[0, 0, :].detach().cpu().numpy().tolist()  # return the cls embedding


class FeatureConverter(nn.Module):
    def __init__(self):
        super(FeatureConverter, self).__init__()
        self.layer1 = nn.Linear(135, 64)
        self.layer2 = nn.Linear(64, 32)

    def forward(self, feature_input):
        hidden_output = self.layer1(feature_input)
        return self.layer2(hidden_output)


class DenseConverter(nn.Module):
    def __init__(self):
        super(DenseConverter, self).__init__()
        self.layer1 = nn.Linear(64, 32)
        self.layer2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, joint_embedding):
        hidden_output = self.layer1(joint_embedding)
        hidden_output2 = self.layer2(hidden_output)
        prob = self.sigmoid(hidden_output2)
        return prob

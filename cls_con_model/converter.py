import torch
import torch.nn as nn
from transformers import AutoModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BertConverter():
    def __init__(self):
        super(BertConverter, self).__init__()
        self.model = AutoModel.from_pretrained('bert-base-german-cased', output_hidden_states=True).to(device)
        self.model.train()

    def forward(self, input_ids, attention_mask):
        bert_results = self.model(input_ids, attention_mask)
        cls = bert_results.last_hidden_state[:, 0, :].flatten()
        return cls.detach().cpu().numpy().tolist()


class FeatureConverter(nn.Module):
    def __init__(self, batch_size):
        super(FeatureConverter, self).__init__()
        self.batch_size = batch_size
        self.layer1 = nn.Linear(24, self.batch_size)
        self.layer2 = nn.Linear(self.batch_size, 128)

    def forward(self, feature_input):
        hidden_output = self.layer1(feature_input)
        return self.layer2(hidden_output).flatten()  # 2048


class DenseConverter(nn.Module):
    def __init__(self, batch_size):
        super(DenseConverter, self).__init__()
        self.batch_size = batch_size
        self.layer1 = nn.Linear(768*self.batch_size + self.batch_size * 128, 64)
        self.layer2 = nn.Linear(64, self.batch_size)
        self.drop = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, joint_embedding):
        hidden_output = self.layer1(joint_embedding)
        hidden_output2 = self.layer2(hidden_output)
        drop_output = self.drop(hidden_output2)
        prob = self.sigmoid(drop_output)
        return prob
import torch.nn as nn

from transformers import AutoModel


class BertConverter:
    def __init__(self):
        super(BertConverter, self).__init__()
        self.model = AutoModel.from_pretrained('bert-base-german-cased')
        self.model.train()  # put the model in eval mode, meaning feed-forward operation?

    def forward(self, input_ids, attention_mask):
        bert_results = self.model(input_ids, attention_mask)
        #print(self.model.training)
        return bert_results.last_hidden_state[0, 0, :].detach().cpu().numpy().tolist()  # return the cls embedding


class FeatureConverter(nn.Module):
    def __init__(self, batch_size):
        super(FeatureConverter, self).__init__()
        self.batch_size = batch_size
        self.layer1 = nn.Linear(3, self.batch_size)
        self.layer2 = nn.Linear(self.batch_size, 32)

    def forward(self, feature_input):
        hidden_output = self.layer1(feature_input)
        return self.layer2(hidden_output).flatten()


class DenseConverter(nn.Module):
    def __init__(self, batch_size):
        super(DenseConverter, self).__init__()
        self.batch_size = batch_size
        self.layer1 = nn.Linear(768+self.batch_size*32, 64)
        self.layer2 = nn.Linear(64, self.batch_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, joint_embedding):
        hidden_output = self.layer1(joint_embedding)
        hidden_output2 = self.layer2(hidden_output)
        prob = self.sigmoid(hidden_output2)
        #print(self.training)
        return prob
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FakeNewsBinaryModel(nn.Module):
    def __init__(self, batch_size):
        super(FakeNewsBinaryModel, self).__init__()
        self.batch_size = batch_size
        self.layer1 = nn.Linear(24, 1)
        #self.layer2 = nn.Linear(64, 1)
        #self.drop = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()


    def forward(self, feature_inputs):
        hidden_output = self.layer1(feature_inputs)
        #hidden_output2 = self.layer2(hidden_output)
        #drop_output = self.drop(hidden_output2)
        prob = self.sigmoid(hidden_output)
        #print(prob)
        return prob



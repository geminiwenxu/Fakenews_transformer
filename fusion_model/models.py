import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


class BertConverter:
    def __init__(self, model_name='bert-base-uncased', device_number=0):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device_number = device_number
        self.model = AutoModel.from_pretrained(model_name).to(device_number)
        self.model.eval()  # put the model in eval mode, meaning feed-forward operation

    def encode(self, sentences, ret_input=False):
        if type(sentences) == str:
            sentences = [sentences]

        for sentence in sentences:
            if len(sentence) > 0 and sentence[-1] != ".":
                sentence += "."

        encoded_input = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True,
                                       max_length=512).to(self.device_number)

        output = self.model(**encoded_input, output_hidden_states=True)
        if ret_input:
            return output, encoded_input
        else:
            return output

    def encode_to_vec(self, sentences):
        bert_results = self.encode(sentences)
        return bert_results.last_hidden_state[0, 0, :].detach().cpu().numpy().tolist()  # return the cls embedding


class BertSentenceConverter:
    def __init__(self, model_name='all-MiniLM-L6-v2', device_number=0):
        self.model = SentenceTransformer(model_name).to(device_number)
        self.model.eval()

    def encode_to_vec(self, sentences, token=None, nlp=False):
        if type(sentences) == str:
            sentences = [sentences]

        for sentence in sentences:
            if len(sentence) > 0 and sentence[-1] != ".":
                sentence += "."

        embeddings = self.model.encode(sentences, convert_to_tensor=True)

        return embeddings.detach().cpu().numpy().tolist()


class FeatureConverter(nn.Module):
    def __init__(self):
        super(FeatureConverter, self).__init__()
        self.layer1 = nn.Linear(135, 64)
        self.layer2 = nn.Linear(64, 32)

    def forward(self, feature_input):
        hidden_output = self.layer1(feature_input)
        return self.layer2(hidden_output)


class DenseModel(nn.Module):
    def __init__(self):
        super(DenseModel, self).__init__()
        self.layer1 = nn.Linear(64, 32)
        self.layer2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fused_embed):
        hidden_output = self.layer1(fused_embed)
        hidden_output2 = self.layer2(hidden_output)
        prob = self.sigmoid(hidden_output2)
        return prob

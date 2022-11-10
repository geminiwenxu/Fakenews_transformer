import torch


def bert_output(
        model,
        data_loader,
        device,
):
    model = model.train()
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    return outputs


def feature_output(model, feature_input):
    model = model.train()
    for i, data in enumerate(feature_input):
        inputs, labels = data
        outputs = model(inputs)
    return outputs


def fusion(bert_model, data_loader, feature_model, feature_input, device):
    model = bert_model.train()
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        bert_output = model(input_ids=input_ids, attention_mask=attention_mask)
    model = feature_model.train()
    for i, data in enumerate(feature_input):
        inputs, labels = data
        feature_outut = model(inputs)

    return torch.cat((bert_output, feature_outut), dim=1)


def train():
    pass
# apply loss and opt to train the dense net 
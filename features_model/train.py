import numpy as np
import torch
import ast
from utilities.weights import weights


def train_epoch(
        model,
        data_loader,
        optimizer,
        device,
        scheduler,
        n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        #input_ids = d["input_ids"].to(device)
        #attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(torch.float32).to(device)
        feature_input = d["feature_input"]
        y = []
        for i in feature_input:
            x = ast.literal_eval(i)
            y.append(x)
        feature_input = torch.tensor(y).to(torch.float32).to(device)
        outputs = model(
            feature_inputs=feature_input
        )
        outputs = outputs.reshape(-1)
        preds = (outputs > 0.5).float()
        class_weights = weights(targets)
        loss_fn = torch.nn.BCELoss(class_weights).to(device)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            #input_ids = d["input_ids"].to(device)
            #attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(torch.float32).to(device)
            feature_input = d["feature_input"]
            y = []
            for i in feature_input:
                x = ast.literal_eval(i)
                y.append(x)
            feature_input = torch.tensor(y).to(torch.float32).to(device)
            outputs = model(
                feature_inputs=feature_input
            )

            outputs = outputs.reshape(-1)
            preds = (outputs > 0.5).float()
            class_weights = weights(targets)
            loss_fn = torch.nn.BCELoss(class_weights).to(device)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

import numpy as np
import torch


def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(torch.float32).to(device)
        re_targets = targets.reshape(len(targets), 1)
        re_targets = re_targets.to(torch.float32)
        feature_input = None

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # feature_inputs=feature_input,
        )
        # print("the final outputs:" ,outputs, outputs.size())
        # print(re_targets, re_targets.size())
        preds = (outputs > 0.45).float()
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        # print("train outputs", outputs)
        # print("train preds", preds)
        # print("train re_targets", re_targets)
        # print("train loss", loss)
        # print("train correct_prediction", correct_predictions)
        losses.append(loss.item())

        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            print(d)
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(torch.float32).to(device)
            re_targets = targets.reshape(len(targets), 1)
            re_targets = re_targets.to(torch.float32)
            feature_input = None
            print('target', targets)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # feature_input=None
            )
            preds = (outputs > 0.45).float()
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            # print("eval outputs", outputs)
            # print("eval preds", preds)
            # print("eval re_targets", re_targets)
            # print("eval loss", loss)
            # print("eval correct_prediction", correct_predictions)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

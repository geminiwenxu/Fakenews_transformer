import ast

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_predictions(model, data_loader):
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["review_text"]
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

            preds = (outputs > 0.5).float()
            probs = outputs

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values

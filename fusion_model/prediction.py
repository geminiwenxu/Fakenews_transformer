import torch
import ast
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
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(torch.float32).to(device)
            feature_input = d["feature_input"]

            y = []
            for i in feature_input:
                x = ast.literal_eval(i)
                y.append(x)
            feature_input = torch.tensor(y).to(torch.float32).to(device)
            print(input_ids.size(), attention_mask.size(), targets.size())
            print("the input", feature_input, type(feature_input), feature_input.size())

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                feature_inputs=feature_input,
            )

            preds = (outputs > 0.45).float()
            # print("pred outputs", outputs)
            # print("preds", preds)

            probs = outputs

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values

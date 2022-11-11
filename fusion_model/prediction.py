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
            sentences = d['sentences'].to(device)
            feature_input = d['features'].to(device)
            label = d['label'].to(device)

            outputs = model(
                sentences,
                feature_input,
            )

            preds = (outputs > 0.45).float()
            # print("pred outputs", outputs)
            # print("preds", preds)

            probs = outputs

            review_texts.extend(sentences)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(label)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values

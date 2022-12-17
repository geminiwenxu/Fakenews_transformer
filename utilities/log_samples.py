import json


def save_samples(y_review_texts, y_pred, y_pred_probs, y_test):
    log_correct = []
    log_wrong = []

    for i in range(0, len(y_pred)):
        correct = dict()
        wrong = dict()
        if y_pred[i] == y_test[i]:
            correct['text'] = y_review_texts[i]
            correct['prob'] = y_pred_probs[i].item()
            correct['prediction'] = y_pred[i].item()
            log_correct.append(correct)

        else:
            wrong['text'] = y_review_texts[i]
            wrong['prob'] = y_pred_probs[i].item()
            wrong['prediction'] = y_pred[i].item()
            log_wrong.append(wrong)

    with open('correct.json', 'w') as c:
        json.dump(log_correct, c)
    with open('wrong.json', 'w') as w:
        json.dump(log_wrong, w)

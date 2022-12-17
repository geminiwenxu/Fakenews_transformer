import torch


def weights(targets):
    num_fake = int((targets == 1.).sum(dim=0))
    num_real = targets.size()[-1]
    w_0 = num_fake / (num_fake + num_real)
    w_1 = num_real / (num_fake + num_real)
    class_weights = []
    for i in targets:
        if i == 0.0:
            class_weights.append(w_0)
        else:
            class_weights.append(w_1)
    return torch.FloatTensor(class_weights)

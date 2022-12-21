import torch


def weights(targets):
    num_fake = int((targets == 1.).sum(dim=0))
    num_real = int((targets == 0.).sum(dim=0))
    class_weights = []
    if num_real != 0 and num_fake != 0:
        w_0 = max(num_fake, num_real) / num_fake
        w_1 = max(num_fake, num_real) / num_real

        for i in targets:
            if i == 0.0:
                class_weights.append(w_1)
            else:
                class_weights.append(w_0)
    else:
        for i in targets:
            class_weights.append(1)

    return torch.FloatTensor(class_weights)

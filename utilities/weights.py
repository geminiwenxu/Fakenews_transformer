import torch
import torch.nn as nn

class_count_df = df.groupby(TARGET).count()
n_0, n_1 = class_count_df.iloc[0, 0], class_count_df.iloc[1, 0]
w_0 = (n_0 + n_1) / (2.0 * n_0)
w_1 = (n_0 + n_1) / (2.0 * n_1)
class_weights = torch.FloatTensor([w_0, w_1]).cuda()

loss_fn = nn.BCELoss(weight=class_weights).to(device)

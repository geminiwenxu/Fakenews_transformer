import torch
from torch.utils.data import Dataset, DataLoader


class ReviewDataset(Dataset):

    def __init__(self, reviews, targets, max_len, feature_input):
        self.reviews = reviews
        self.targets = targets
        # self.tokenizer = tokenizer
        self.max_len = max_len
        self.feature_input = feature_input

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        feature_input = self.feature_input[item]
        return {
            'review_text': review,
            'targets': torch.tensor(target, dtype=torch.long),
            'feature_input': feature_input
        }


def create_data_loader(df, max_len, batch_size):
    ds = ReviewDataset(
        reviews=df.text.to_numpy(),
        targets=df.label_id.to_numpy(),
        max_len=max_len,
        feature_input=df.feature_input
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True
    )

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased', do_lower_case=False)


class ReviewDataset(Dataset):

    def __init__(self, reviews, targets, tokenizer, max_len, feature_input):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.feature_input = feature_input

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        feature_input = self.feature_input[item]
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        length = encoding.attention_mask.sum(1)
        return {
            'length': torch.tensor(length),
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long),
            'feature_input': feature_input
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = ReviewDataset(
        reviews=df.text.to_numpy(),
        targets=df.label_id.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
        feature_input=df.feature_input
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True
    )

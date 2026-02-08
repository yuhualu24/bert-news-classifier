import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding


class BBCDataset(Dataset):
    """PyTorch Dataset that pairs BERT tokenizer encodings with integer labels."""

    def __init__(self, encodings: BatchEncoding, labels: list[int]):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
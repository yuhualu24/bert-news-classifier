import logging

from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from data.text_dataset import TextClassificationDataset

logger = logging.getLogger(__name__)

REUTERS_LABEL_NAMES = [
    "acq", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade"
]


class ReutersPreprocessor:
    """
    Preprocesses the Reuters-21578 (8-class) dataset from HuggingFace
    for BERT classification.

    This dataset comes with pre-defined train/test splits and integer labels,
    so no manual splitting or label encoding is needed.

    Usage:
        preprocessor = ReutersPreprocessor()
        train_loader, val_loader = preprocessor.run()
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 256,
        batch_size: int = 16,
        max_samples: int | None = None,
    ):
        self.max_length = max_length
        self.batch_size = batch_size
        self.max_samples = max_samples

        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def run(self) -> tuple[DataLoader, DataLoader]:
        """Download -> (optionally limit) -> tokenize -> DataLoaders."""
        ds = load_dataset("yangwang825/reuters-21578")

        train_ds = ds["train"]
        val_ds = ds["test"]

        if self.max_samples is not None:
            train_ds = train_ds.shuffle(seed=42).select(
                range(min(self.max_samples, len(train_ds)))
            )
            val_ds = val_ds.shuffle(seed=42).select(
                range(min(self.max_samples, len(val_ds)))
            )

        train_texts = list(train_ds["text"])
        train_labels = list(train_ds["label"])
        val_texts = list(val_ds["text"])
        val_labels = list(val_ds["label"])

        logger.info(
            "Loaded Reuters-21578 — %d train / %d val samples across %s",
            len(train_texts), len(val_texts), REUTERS_LABEL_NAMES,
        )

        train_enc = self.tokenizer(
            train_texts, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        val_enc = self.tokenizer(
            val_texts, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )

        train_loader = DataLoader(
            TextClassificationDataset(train_enc, train_labels),
            batch_size=self.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TextClassificationDataset(val_enc, val_labels),
            batch_size=self.batch_size,
        )

        logger.info("Ready -> %d train / %d val batches", len(train_loader), len(val_loader))
        return train_loader, val_loader

    @property
    def num_labels(self) -> int:
        return len(REUTERS_LABEL_NAMES)

    @property
    def label_names(self) -> list[str]:
        return REUTERS_LABEL_NAMES

import logging

from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from data.bbc_dataset import BBCDataset

logger = logging.getLogger(__name__)

AG_NEWS_LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


class AGNewsPreprocessor:
    """
    Preprocesses the AG News dataset (from HuggingFace) for BERT classification.

    Usage:
        preprocessor = AGNewsPreprocessor()
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
        ds = load_dataset("ag_news")

        train_ds = ds["train"]
        val_ds = ds["test"]

        if self.max_samples is not None:
            train_ds = train_ds.shuffle(seed=42).select(range(min(self.max_samples, len(train_ds))))
            val_ds = val_ds.shuffle(seed=42).select(range(min(self.max_samples, len(val_ds))))

        train_texts = list(train_ds["text"])
        train_labels = list(train_ds["label"])
        val_texts = list(val_ds["text"])
        val_labels = list(val_ds["label"])

        logger.info(
            "Loaded AG News — %d train / %d val samples across %s",
            len(train_texts), len(val_texts), AG_NEWS_LABEL_NAMES,
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
            BBCDataset(train_enc, train_labels), batch_size=self.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            BBCDataset(val_enc, val_labels), batch_size=self.batch_size,
        )

        logger.info("Ready -> %d train / %d val batches", len(train_loader), len(val_loader))
        return train_loader, val_loader

    @property
    def num_labels(self) -> int:
        return len(AG_NEWS_LABEL_NAMES)

    @property
    def label_names(self) -> list[str]:
        return AG_NEWS_LABEL_NAMES

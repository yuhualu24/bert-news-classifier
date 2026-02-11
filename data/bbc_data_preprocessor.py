import os
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from data.text_dataset import TextClassificationDataset

logger = logging.getLogger(__name__)


class BBCDataPreprocessor:
    """
    Preprocesses the BBC News dataset for BERT classification.

    Usage:
        preprocessor = BBCDataPreprocessor("path/to/bbc")
        train_loader, val_loader = preprocessor.run()
    """

    def __init__(
        self,
        bbc_data_folder_path: str,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        test_size: float = 0.2,
        batch_size: int = 16,
    ):
        self.data_path = bbc_data_folder_path
        self.max_length = max_length
        self.test_size = test_size
        self.batch_size = batch_size

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()

    def load_raw_data(self) -> tuple[list[str], list[str]]:
        """Read all .txt files from the category-folder structure."""
        texts, labels = [], []
        for category in sorted(os.listdir(self.data_path)):
            category_dir = os.path.join(self.data_path, category)
            if not os.path.isdir(category_dir):
                continue
            for fname in os.listdir(category_dir):
                with open(os.path.join(category_dir, fname), encoding="utf-8", errors="replace") as f:
                    texts.append(f.read())
                    labels.append(category)

        logger.info("Loaded %d articles across %s", len(texts), sorted(set(labels)))
        return texts, labels

    def run(self) -> tuple[DataLoader, DataLoader]:
        """Load -> encode -> split -> tokenize -> DataLoaders."""
        texts, labels = self.load_raw_data()
        labels_encoded = self.label_encoder.fit_transform(labels).tolist()

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels_encoded,
            test_size=self.test_size, stratify=labels_encoded, random_state=42,
        )

        train_enc = self.tokenizer(train_texts, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        val_enc = self.tokenizer(val_texts, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")

        train_loader = DataLoader(TextClassificationDataset(train_enc, train_labels), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TextClassificationDataset(val_enc, val_labels), batch_size=self.batch_size)

        logger.info("Ready -> %d train / %d val samples", len(train_labels), len(val_labels))
        return train_loader, val_loader

    @property
    def num_labels(self) -> int:
        return len(self.label_encoder.classes_)

    @property
    def label_names(self) -> list[str]:
        return list(self.label_encoder.classes_)
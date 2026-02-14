import logging

from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from data.text_dataset import TextClassificationDataset
from data.preprocessor_result import PreprocessorResult

logger = logging.getLogger(__name__)

BBC_LABEL_NAMES = ["business", "entertainment", "politics", "sport", "tech"]


class BBCDataPreprocessor:
    """
    Preprocesses the BBC News dataset for BERT classification.

    Loads the dataset from HuggingFace (SetFit/bbc-news) which provides
    pre-defined train/test splits with 5 categories.

    Usage:
        preprocessor = BBCDataPreprocessor()
        result = preprocessor.run()
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        batch_size: int = 16,
    ):
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def run(self) -> PreprocessorResult:
        """Load from HuggingFace -> tokenize -> DataLoaders."""
        ds = load_dataset("SetFit/bbc-news")

        train_split = ds["train"]
        val_split = ds["test"]

        train_texts = train_split["text"]
        train_labels = train_split["label"]
        val_texts = val_split["text"]
        val_labels = val_split["label"]
        val_label_names_str = val_split["label_text"]

        logger.info(
            "Loaded BBC News — %d train / %d val samples across %d categories",
            len(train_texts), len(val_texts), len(BBC_LABEL_NAMES),
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
        return PreprocessorResult(
            train_loader=train_loader,
            val_loader=val_loader,
            val_texts=val_texts,
            val_labels_encoded=val_labels,
            val_label_names_str=val_label_names_str,
            label_names=BBC_LABEL_NAMES,
            num_labels=len(BBC_LABEL_NAMES),
        )

    @property
    def num_labels(self) -> int:
        return len(BBC_LABEL_NAMES)

    @property
    def label_names(self) -> list[str]:
        return BBC_LABEL_NAMES
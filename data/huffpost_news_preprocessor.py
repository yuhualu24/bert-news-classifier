import logging
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from data.text_dataset import TextClassificationDataset

logger = logging.getLogger(__name__)


class HuffPostNewsPreprocessor:

    def __init__(
            self,
            model_name: str = "bert-base-uncased",
            max_length: int = 256,
            test_size: float = 0.2,
            batch_size: int = 16,
            max_samples: int | None = None,
    ):
        self.max_length = max_length
        self.test_size = test_size
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()

    def run(self) -> tuple[DataLoader, DataLoader]:

        ds = load_dataset("heegyu/news-category-dataset")
        full_ds = ds["train"]

        full_ds = self.merge(full_ds)

        if self.max_samples is not None:
            full_ds = full_ds.shuffle(seed=42).select(range(min(self.max_samples, len(full_ds))))

        # extract texts and labels from the dataset
        texts = [
            (h + " " + d).strip() if d else h
            for h, d in zip(full_ds["headline"], full_ds["short_description"])
        ]
        labels = list(full_ds["category"])
        labels_encoded = self.label_encoder.fit_transform(labels).tolist()

        # stratified split to keep category proportions in train and val
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels_encoded,
            test_size=self.test_size, stratify=labels_encoded, random_state=42,
        )

        logger.info("Loaded HuffPost News — %d train / %d val samples across %d categories",
                    len(train_texts), len(val_texts), len(self.label_encoder.classes_))

        train_enc = self.tokenizer(
            train_texts, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )

        val_enc = self.tokenizer(
            val_texts, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )

        train_loader = DataLoader(
            TextClassificationDataset(train_enc, train_labels), batch_size=self.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TextClassificationDataset(val_enc, val_labels), batch_size=self.batch_size,
        )

        logger.info("Ready -> %d train / %d val batches", len(train_loader), len(val_loader))
        return train_loader, val_loader

    def merge(self, ds):

        # manually merge similar categories
        CATEGORY_MAPPING = {
            # Duplicates — exact same topic, different names
            "PARENTS": "PARENTING",
            "STYLE": "STYLE & BEAUTY",
            "TASTE": "FOOD & DRINK",
            "THE WORLDPOST": "WORLD NEWS",
            "WORLDPOST": "WORLD NEWS",
            "GREEN": "ENVIRONMENT",
            "ENVIRONMENT": "ENVIRONMENT",  # normalize from ENVIRO if present
            "HEALTHY LIVING": "WELLNESS",
            "ARTS & CULTURE": "ARTS",

            # Thematic merges — closely related topics
            # "COMEDY": "ENTERTAINMENT",
            # "WEIRD NEWS": "ENTERTAINMENT",
            # "FIFTY": "LIFESTYLE",
            # "HOME & LIVING": "LIFESTYLE",
            # "WEDDINGS": "LIFESTYLE",
            # "DIVORCE": "LIFESTYLE",
            # "GOOD NEWS": "MISCELLANEOUS",
            # "IMPACT": "MISCELLANEOUS",
        }

        def remap_category(category):
            return CATEGORY_MAPPING.get(category, category)

        return ds.map(lambda x: {"category": remap_category(x["category"])})

    @property
    def num_labels(self) -> int:
        return len(self.label_encoder.classes_)

    @property
    def label_names(self) -> list[str]:
        return list(self.label_encoder.classes_)


# if __name__ == "__main__":
#
#     preprocessor = HuffPostNewsPreprocessor()
#     train_loader, val_loader = preprocessor.run()
#     print(len(train_loader), len(val_loader))

import logging

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from config import Config
from model.classifier import BertClassifier

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Generates detailed evaluation metrics for a trained BBCBertClassifier.

    Produces per-class precision/recall/F1 and a confusion matrix.
    """

    def __init__(
        self,
        model: BertClassifier,
        config: Config,
        device: torch.device | None = None,
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        """
        Run the model over an entire DataLoader and return:
            - classification_report (str and dict)
            - confusion_matrix
            - overall accuracy
        """
        self.model.eval()
        all_preds, all_labels = [], []

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            preds = self.model.predict(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

        target_names = self.config.label_names or None
        labels = list(range(len(target_names))) if target_names else None

        report_str = classification_report(
            all_labels, all_preds, target_names=target_names, labels=labels
        )
        report_dict = classification_report(
            all_labels, all_preds, target_names=target_names, labels=labels, output_dict=True
        )
        cm = confusion_matrix(all_labels, all_preds, labels=labels)

        logger.info("Classification Report:\n%s", report_str)
        logger.info("Confusion Matrix:\n%s", cm)

        return {
            "report_str": report_str,
            "report_dict": report_dict,
            "confusion_matrix": cm,
            "accuracy": report_dict["accuracy"],
        }
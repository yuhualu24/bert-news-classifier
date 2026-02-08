import os
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import BertModel

from config import Config


logger = logging.getLogger(__name__)


@dataclass
class ClassifierOutput:
    """Simple container to mirror HuggingFace's SequenceClassifierOutput."""
    loss: torch.Tensor | None
    logits: torch.Tensor


class BBCBertClassifier(nn.Module):
    """
    Base BertModel for feature extraction with a custom linear classification head.

    Architecture:
        BertModel -> [CLS] token hidden state -> Dropout -> Linear -> logits

    The entire model (BERT + linear head) is fine-tuned end-to-end.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> ClassifierOutput:
        """
        Forward pass through base BERT + linear classifier.

        Returns:
            ClassifierOutput with .loss and .logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Extract the [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)

        logits = self.classifier(self.dropout(cls_output))  # (batch, num_labels)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return ClassifierOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def predict(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return predicted class indices for a batch."""
        self.eval()
        output = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits.argmax(dim=-1)

    def save(self, path: str | None = None) -> str:
        """Save BERT weights and classifier head to disk."""
        save_dir = path or os.path.join(self.config.output_dir, "model")
        os.makedirs(save_dir, exist_ok=True)

        self.bert.save_pretrained(save_dir)
        torch.save(self.classifier.state_dict(), os.path.join(save_dir, "classifier_head.pt"))

        logger.info("Model saved to %s", save_dir)
        return save_dir

    @classmethod
    def load(cls, path: str, config: Config) -> "BBCBertClassifier":
        """Load a previously saved model."""
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.config = config

        instance.bert = BertModel.from_pretrained(path)
        instance.dropout = nn.Dropout(0.1)
        instance.classifier = nn.Linear(config.hidden_size, config.num_labels)
        instance.classifier.load_state_dict(
            torch.load(os.path.join(path, "classifier_head.pt"), weights_only=True)
        )
        instance.loss_fn = nn.CrossEntropyLoss()

        logger.info("Model loaded from %s", path)
        return instance

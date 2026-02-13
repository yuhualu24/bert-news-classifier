import copy
import logging
from dataclasses import dataclass

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import Config
from model.classifier import BertClassifier

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float


class Trainer:
    """
    Fine-tuning loop for BBCBertClassifier.

    Handles optimizer setup, epoch iteration, and basic validation.
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

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.history: list[EpochMetrics] = []

    # ------------------------------------------------------------------
    # Single training epoch
    # ------------------------------------------------------------------
    def _train_one_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    # ------------------------------------------------------------------
    # Single validation pass
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            total_loss += outputs.loss.item()

            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    # Full training run
    # ------------------------------------------------------------------
    def train(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> list[EpochMetrics]:
        """Run the training loop with early stopping.

        Tracks the best val_loss and restores the best model weights
        if training is stopped early or completes all epochs.
        """
        patience = self.config.early_stopping_patience
        logger.info(
            "Starting training for up to %d epochs on %s (early stopping patience=%d)",
            self.config.num_epochs, self.device, patience,
        )

        best_val_loss = float("inf")
        best_model_state = None
        epochs_without_improvement = 0

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_loss, val_acc = self._validate(val_loader)

            metrics = EpochMetrics(epoch, train_loss, val_loss, val_acc)
            self.history.append(metrics)

            logger.info(
                "Epoch %d/%d — train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
                epoch,
                self.config.num_epochs,
                train_loss,
                val_loss,
                val_acc,
            )

            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
                logger.info("  ↳ New best val_loss=%.4f — saving model weights", val_loss)
            else:
                epochs_without_improvement += 1
                logger.info(
                    "  ↳ No improvement for %d epoch(s) (best val_loss=%.4f)",
                    epochs_without_improvement, best_val_loss,
                )

            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping triggered after %d epochs (patience=%d)",
                    epoch, patience,
                )
                break

        # Restore best model weights
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Restored best model weights (val_loss=%.4f)", best_val_loss)

        return self.history
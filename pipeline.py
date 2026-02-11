import logging
import platform
import sys
import time
import torch

from data.bbc_data_preprocessor import BBCDataPreprocessor
from data.ag_news_preprocessor import AGNewsPreprocessor
from data.huffpost_news_preprocessor import HuffPostNewsPreprocessor
from data.reuters_preprocessor import ReutersPreprocessor
from model.classifier import BertClassifier
from test.trainer import Trainer
from test.evaluator import Evaluator
from config import Config

logger = logging.getLogger(__name__)


def _log_environment() -> None:
    """Log system / library versions and device info."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lines = [
        "=== Environment ===",
        f"  Python:       {sys.version}",
        f"  Platform:     {platform.platform()}",
        f"  PyTorch:      {torch.__version__}",
        f"  CUDA avail:   {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"  GPU:          {torch.cuda.get_device_name(0)}")
    lines.append(f"  Device:       {device}")
    logger.info("\n".join(lines))


def _log_config(config: Config) -> None:
    """Log all hyperparameters from Config."""
    lines = ["=== Config ==="]
    for field_name, value in vars(config).items():
        lines.append(f"  {field_name}: {value}")
    logger.info("\n".join(lines))


def run_pipeline(
    dataset_name: str = "bbc",
    data_path: str | None = None,
    num_epochs: int = 3,
    max_samples: int | None = None,
) -> dict:
    """Run the full train & evaluate pipeline. Returns evaluation results."""
    t_start = time.time()

    # 1. Preprocess
    config = Config(dataset_name=dataset_name, num_epochs=num_epochs, max_samples=max_samples)

    _log_environment()
    _log_config(config)

    logger.info("=== Data Preprocessing ===")
    if dataset_name == "bbc":
        if data_path is None:
            raise ValueError("data_path is required for the BBC dataset")
        preprocessor = BBCDataPreprocessor(
            data_path,
            model_name=config.model_name,
            max_length=config.max_length,
            test_size=config.test_size,
            batch_size=config.batch_size,
        )
    elif dataset_name == "ag_news":
        preprocessor = AGNewsPreprocessor(
            model_name=config.model_name,
            max_length=config.max_length,
            batch_size=config.batch_size,
            max_samples=config.max_samples,
        )
    elif dataset_name == "huffpost_news":
        preprocessor = HuffPostNewsPreprocessor(
            model_name=config.model_name,
            max_length=config.max_length,
            batch_size=config.batch_size,
            max_samples=config.max_samples,
        )
    elif dataset_name == "reuters":
        preprocessor = ReutersPreprocessor(
            model_name=config.model_name,
            max_length=config.max_length,
            batch_size=config.batch_size,
            max_samples=config.max_samples,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'bbc', 'ag_news', 'huffpost_news', or 'reuters'.")

    train_loader, val_loader = preprocessor.run()
    logger.info("Data preprocessing completed successfully")

    # 2. Build model
    config.num_labels = preprocessor.num_labels
    model = BertClassifier(config=config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "=== Model ===\n  Architecture: BERT + Linear\n  Total params:     %s\n  Trainable params: %s",
        f"{total_params:,}", f"{trainable_params:,}",
    )

    # 3. Train
    logger.info("=== Training ===")
    trainer = Trainer(model, config=config)
    trainer.train(train_loader, val_loader)

    # 4. Evaluate
    logger.info("=== Evaluation ===")
    config.label_names = preprocessor.label_names
    evaluator = Evaluator(model, config=config)
    results = evaluator.evaluate(val_loader)

    # 5. Save
    save_path = model.save("outputs/model")

    elapsed = time.time() - t_start
    mins, secs = divmod(int(elapsed), 60)
    hrs, mins = divmod(mins, 60)
    logger.info("=== Run Complete ===\n  Total time: %dh %dm %ds\n  Model saved to: %s",
                hrs, mins, secs, save_path)

    return results

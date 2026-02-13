import json
import logging
import os
import platform
import sys
import time
from datetime import datetime

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
    """Run the full train & evaluate pipeline.

    Saves all artifacts to outputs/runs/{dataset_name}_{timestamp}/ so that
    compare.py can later load them for Gemini evaluation.

    Returns evaluation results dict including the run_dir path.
    """
    t_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    prep_result = preprocessor.run()
    train_loader = prep_result.train_loader
    val_loader = prep_result.val_loader
    logger.info("Data preprocessing completed successfully")

    # 2. Build model
    config.num_labels = prep_result.num_labels
    config.label_names = prep_result.label_names
    model = BertClassifier(config=config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "=== Model ===\n  Architecture: BERT + Linear\n  Total params:     %s\n  Trainable params: %s",
        f"{total_params:,}", f"{trainable_params:,}",
    )

    # 3. Train
    logger.info("=== Training ===")
    t_train_start = time.time()
    trainer = Trainer(model, config=config)
    trainer.train(train_loader, val_loader)
    train_time = time.time() - t_train_start

    # 4. Evaluate
    logger.info("=== Evaluation ===")
    t_eval_start = time.time()
    evaluator = Evaluator(model, config=config)
    results = evaluator.evaluate(val_loader)
    eval_time = time.time() - t_eval_start

    elapsed = time.time() - t_start

    # 5. Save all artifacts to run directory
    run_dir = os.path.join("outputs", "runs", f"{dataset_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info("=== Saving Run Artifacts to %s ===", run_dir)

    # 5a. Model weights
    model_dir = os.path.join(run_dir, "model")
    model.save(model_dir)

    # 5b. Run metadata
    metadata = {
        "dataset_name": dataset_name,
        "model_name": config.model_name,
        "num_labels": prep_result.num_labels,
        "label_names": prep_result.label_names,
        "max_samples": max_samples,
        "test_size": config.test_size,
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "num_epochs": num_epochs,
        "early_stopping_patience": config.early_stopping_patience,
        "timestamp": timestamp,
        "total_train_samples": len(train_loader.dataset),
        "total_val_samples": len(val_loader.dataset),
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # 5c. BERT evaluation metrics
    bert_metrics = {
        "accuracy": results["accuracy"],
        "macro_f1": results["report_dict"]["macro avg"]["f1-score"],
        "train_time_s": train_time,
        "eval_time_s": eval_time,
        "total_time_s": elapsed,
        "report_dict": results["report_dict"],
    }
    with open(os.path.join(run_dir, "bert_eval_metrics.json"), "w") as f:
        json.dump(bert_metrics, f, indent=2)

    # 5d. Human-readable classification report
    with open(os.path.join(run_dir, "bert_classification_report.txt"), "w") as f:
        f.write(f"BERT Classification Report — {dataset_name}\n")
        f.write(f"Model: {config.model_name}\n")
        f.write(f"Samples: {len(val_loader.dataset)}\n\n")
        f.write(results["report_str"])

    # 5e. Validation data (for later Gemini evaluation)
    val_data = {
        "texts": prep_result.val_texts,
        "labels_encoded": prep_result.val_labels_encoded,
        "label_names_str": prep_result.val_label_names_str,
    }
    with open(os.path.join(run_dir, "val_data.json"), "w") as f:
        json.dump(val_data, f)

    # 5f. Training history
    history = [
        {
            "epoch": m.epoch,
            "train_loss": m.train_loss,
            "val_loss": m.val_loss,
            "val_accuracy": m.val_accuracy,
        }
        for m in trainer.history
    ]
    with open(os.path.join(run_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    mins, secs = divmod(int(elapsed), 60)
    hrs, mins = divmod(mins, 60)
    logger.info("=== Run Complete ===\n  Total time: %dh %dm %ds\n  Run saved to: %s",
                hrs, mins, secs, run_dir)

    results["run_dir"] = run_dir
    return results

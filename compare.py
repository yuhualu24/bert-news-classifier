"""
BERT vs Gemini 2.5 Flash comparison on any supported news dataset.

Supports: bbc, ag_news, huffpost_news, reuters.

Runs both models on the same validation set and logs accuracy, macro-F1,
time, and cost side by side.

Usage:
    python compare.py
"""

import logging
import os
import time
from datetime import datetime

import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from config import Config
from model.classifier import BertClassifier
from test.trainer import Trainer
from gemini.classifier import GeminiClassifier
from gemini.batch_runner import GeminiBatchRunner
from data.ag_news_preprocessor import AG_NEWS_LABEL_NAMES
from data.reuters_preprocessor import REUTERS_LABEL_NAMES

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logging() -> str:
    """Configure console + file logging. Returns the log file path."""
    os.makedirs("outputs/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"outputs/logs/compare_{timestamp}.txt"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(file_handler)

    return log_path


def load_and_split_data(config: Config, data_path: str | None = None):
    """
    Load any supported dataset and split into train/val.

    Supports: "bbc", "ag_news", "huffpost_news", "reuters".

    Returns (train_texts, val_texts, train_labels_encoded, val_labels_encoded,
             val_label_names_str, label_names_list).
    """
    dataset_name = config.dataset_name

    if dataset_name == "bbc":
        if data_path is None:
            raise ValueError("data_path is required for the BBC dataset")
        texts, labels = [], []
        for category in sorted(os.listdir(data_path)):
            category_dir = os.path.join(data_path, category)
            if not os.path.isdir(category_dir):
                continue
            for fname in os.listdir(category_dir):
                with open(os.path.join(category_dir, fname), encoding="utf-8", errors="replace") as f:
                    texts.append(f.read())
                    labels.append(category)

        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels).tolist()
        label_names = list(label_encoder.classes_)

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels_encoded,
            test_size=config.test_size, stratify=labels_encoded, random_state=42,
        )
        val_label_names = [label_encoder.inverse_transform([idx])[0] for idx in val_labels]

    elif dataset_name == "ag_news":
        ds = load_dataset("ag_news")
        train_ds, val_ds = ds["train"], ds["test"]

        if config.max_samples is not None:
            train_ds = train_ds.shuffle(seed=42).select(range(min(config.max_samples, len(train_ds))))
            val_ds = val_ds.shuffle(seed=42).select(range(min(config.max_samples, len(val_ds))))

        train_texts = list(train_ds["text"])
        train_labels = list(train_ds["label"])
        val_texts = list(val_ds["text"])
        val_labels = list(val_ds["label"])
        label_names = list(AG_NEWS_LABEL_NAMES)
        val_label_names = [label_names[idx] for idx in val_labels]

    elif dataset_name == "huffpost_news":
        ds = load_dataset("heegyu/news-category-dataset")
        full_ds = ds["train"]

        if config.max_samples is not None:
            full_ds = full_ds.shuffle(seed=42).select(
                range(min(config.max_samples, len(full_ds)))
            )

        texts = [
            (h + " " + d).strip() if d else h
            for h, d in zip(full_ds["headline"], full_ds["short_description"])
        ]
        labels = list(full_ds["category"])

        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels).tolist()
        label_names = list(label_encoder.classes_)

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels_encoded,
            test_size=config.test_size, stratify=labels_encoded, random_state=42,
        )
        val_label_names = [label_encoder.inverse_transform([idx])[0] for idx in val_labels]

    elif dataset_name == "reuters":
        ds = load_dataset("yangwang825/reuters-21578")
        train_ds, val_ds = ds["train"], ds["test"]

        if config.max_samples is not None:
            train_ds = train_ds.shuffle(seed=42).select(range(min(config.max_samples, len(train_ds))))
            val_ds = val_ds.shuffle(seed=42).select(range(min(config.max_samples, len(val_ds))))

        train_texts = list(train_ds["text"])
        train_labels = list(train_ds["label"])
        val_texts = list(val_ds["text"])
        val_labels = list(val_ds["label"])
        label_names = list(REUTERS_LABEL_NAMES)
        val_label_names = [label_names[idx] for idx in val_labels]

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            "Use 'bbc', 'ag_news', 'huffpost_news', or 'reuters'."
        )

    logger.info(
        "Loaded %s — %d train / %d val samples, %d categories",
        dataset_name, len(train_texts), len(val_texts), len(label_names),
    )

    return train_texts, val_texts, train_labels, val_labels, val_label_names, label_names


def run_bert(
    train_texts, val_texts, train_labels, val_labels,
    label_names, config,
) -> dict:
    """Train BERT and evaluate on the val set. Returns metrics + timing."""
    from transformers import BertTokenizer
    from data.text_dataset import TextClassificationDataset
    from torch.utils.data import DataLoader

    tokenizer = BertTokenizer.from_pretrained(config.model_name)

    logger.info("=== BERT: Tokenizing ===")
    train_enc = tokenizer(
        train_texts, truncation=True, padding="max_length",
        max_length=config.max_length, return_tensors="pt",
    )
    val_enc = tokenizer(
        val_texts, truncation=True, padding="max_length",
        max_length=config.max_length, return_tensors="pt",
    )

    train_loader = DataLoader(
        TextClassificationDataset(train_enc, train_labels),
        batch_size=config.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TextClassificationDataset(val_enc, val_labels),
        batch_size=config.batch_size,
    )

    # Build model
    config.num_labels = len(label_names)
    model = BertClassifier(config=config)

    # Train
    logger.info("=== BERT: Training ===")
    t_start = time.time()
    trainer = Trainer(model, config=config)
    trainer.train(train_loader, val_loader)
    train_time = time.time() - t_start

    # Evaluate
    logger.info("=== BERT: Evaluating ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    t_eval_start = time.time()
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model.predict(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
    eval_time = time.time() - t_eval_start

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    labels_range = list(range(len(label_names)))
    report = classification_report(
        all_labels, all_preds, target_names=label_names, labels=labels_range
    )
    logger.info("BERT Classification Report:\n%s", report)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "train_time_s": train_time,
        "eval_time_s": eval_time,
        "total_time_s": train_time + eval_time,
        "cost_usd": 0.0,  # Local compute
        "report": report,
    }


def run_gemini(val_texts, val_label_names, label_names, delay_seconds=1.0) -> dict:
    """Run Gemini zero-shot classification on the val set. Returns metrics + timing."""

    logger.info("=== Gemini: Classifying ===")
    classifier = GeminiClassifier(label_names=label_names)
    runner = GeminiBatchRunner(classifier, delay_seconds=delay_seconds)

    results = runner.run(val_texts, val_label_names)

    # Compute metrics (string-based comparison)
    preds = results["predictions"]
    trues = results["true_labels"]

    # For accuracy/F1, treat UNKNOWN as a wrong prediction
    accuracy = accuracy_score(trues, preds)
    macro_f1 = f1_score(trues, preds, average="macro", zero_division=0)

    report = classification_report(
        trues, preds, labels=label_names, zero_division=0,
    )
    logger.info("Gemini Classification Report:\n%s", report)

    unknown_count = sum(1 for p in preds if p == "UNKNOWN")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "train_time_s": 0.0,  # No training
        "eval_time_s": results["elapsed_seconds"],
        "total_time_s": results["elapsed_seconds"],
        "cost_usd": results["cost_usd"],
        "input_tokens": results["total_input_tokens"],
        "output_tokens": results["total_output_tokens"],
        "unknown_count": unknown_count,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Comparison output
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    hrs, remainder = divmod(int(seconds), 3600)
    mins, secs = divmod(remainder, 60)
    if hrs > 0:
        return f"{hrs}h {mins}m {secs}s"
    elif mins > 0:
        return f"{mins}m {secs}s"
    else:
        return f"{secs}s"


def log_comparison(bert_results: dict, gemini_results: dict, num_samples: int) -> str:
    """Format and return a comparison summary string."""
    sep = "=" * 70
    gemini_cost_str = f"${gemini_results['cost_usd']:.4f}"
    lines = [
        sep,
        "BERT vs Gemini 2.5 Flash — Comparison Results",
        sep,
        f"Evaluation samples: {num_samples}",
        "",
        f"{'Metric':<20} {'BERT':>15} {'Gemini 2.5 Flash':>20}",
        f"{'-'*20} {'-'*15} {'-'*20}",
        f"{'Accuracy':<20} {bert_results['accuracy']:>15.4f} {gemini_results['accuracy']:>20.4f}",
        f"{'Macro-F1':<20} {bert_results['macro_f1']:>15.4f} {gemini_results['macro_f1']:>20.4f}",
        f"{'Train time':<20} {format_time(bert_results['train_time_s']):>15} {'N/A (zero-shot)':>20}",
        f"{'Eval time':<20} {format_time(bert_results['eval_time_s']):>15} {format_time(gemini_results['eval_time_s']):>20}",
        f"{'Total time':<20} {format_time(bert_results['total_time_s']):>15} {format_time(gemini_results['total_time_s']):>20}",
        f"{'Cost (USD)':<20} {'$0.00 (local)':>15} {gemini_cost_str:>20}",
        "",
    ]

    if "unknown_count" in gemini_results:
        lines.append(f"Gemini unmatched responses: {gemini_results['unknown_count']}/{num_samples}")
        lines.append("")

    if "input_tokens" in gemini_results:
        lines.append(f"Gemini token usage: {gemini_results['input_tokens']:,} input / {gemini_results['output_tokens']:,} output")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log_path = setup_logging()
    logger.info("Comparison log: %s", log_path)

    # --- Configuration ---
    DATASET_NAME = "reuters"  # Options: "bbc", "ag_news", "huffpost_news", "reuters"
    DATA_PATH = None                # Required only for "bbc" (e.g. "bbc")
    MAX_SAMPLES = 2500              # Total dataset samples (train + val)
    NUM_EPOCHS = 3
    GEMINI_DELAY = 1.0              # Seconds between Gemini API calls

    config = Config(
        dataset_name=DATASET_NAME,
        num_epochs=NUM_EPOCHS,
        max_samples=MAX_SAMPLES,
    )

    # --- Load data once (shared between BERT and Gemini) ---
    logger.info("=== Loading Data (%s) ===", DATASET_NAME)
    (
        train_texts, val_texts,
        train_labels, val_labels,
        val_label_names, label_names,
    ) = load_and_split_data(config, data_path=DATA_PATH)

    # --- Run BERT ---
    bert_results = run_bert(
        train_texts, val_texts, train_labels, val_labels,
        label_names, config,
    )

    # --- Run Gemini ---
    gemini_results = run_gemini(
        val_texts, val_label_names, label_names,
        delay_seconds=GEMINI_DELAY,
    )

    # --- Compare ---
    summary = log_comparison(bert_results, gemini_results, len(val_texts))
    logger.info("\n%s", summary)

    # Also save comparison to a dedicated file
    os.makedirs("outputs", exist_ok=True)
    comparison_path = f"outputs/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(comparison_path, "w") as f:
        f.write(summary)
        f.write("\n\n--- BERT Classification Report ---\n")
        f.write(bert_results["report"])
        f.write("\n\n--- Gemini Classification Report ---\n")
        f.write(gemini_results["report"])

    logger.info("Comparison saved to %s", comparison_path)
    logger.info("Full log saved to %s", log_path)

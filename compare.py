"""
BERT vs Gemini 2.5 Flash comparison on the same validation data.

Loads BERT results from a previous training run (python main.py) and runs
Gemini evaluation with checkpoint/resume support for daily API limits.

Usage:
    # Auto-detect latest run for huffpost_news:
    python compare.py

    # Specify a run directory:
    python compare.py --run-dir outputs/runs/huffpost_news_20260213_143022

    # Override daily limit (useful for testing):
    python compare.py --daily-limit 100

    # Resume an interrupted Gemini run (automatic — just re-run):
    python compare.py --run-dir outputs/runs/huffpost_news_20260213_143022
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import Config
from gemini.classifier import GeminiClassifier
from gemini.batch_runner import GeminiBatchRunner

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logging(dataset_name: str) -> str:
    """Configure console + file logging. Returns the log file path."""
    os.makedirs("outputs/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"outputs/logs/compare_bert_{dataset_name}_{timestamp}.txt"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(file_handler)

    return log_path


def find_latest_run(dataset_name: str, base_dir: str = "outputs/runs") -> str:
    """Find the most recent run directory for a given dataset name."""
    runs_path = Path(base_dir)
    if not runs_path.exists():
        raise FileNotFoundError(f"No runs directory found at {base_dir}")

    matching = sorted(
        [d for d in runs_path.iterdir()
         if d.is_dir() and d.name.startswith(dataset_name + "_")],
        key=lambda d: d.name,
        reverse=True,
    )
    if not matching:
        raise FileNotFoundError(
            f"No runs found for dataset '{dataset_name}' in {base_dir}. "
            "Run 'python main.py' first to train BERT."
        )
    return str(matching[0])


def load_bert_run(run_dir: str) -> dict:
    """Load all BERT run artifacts from a run directory."""
    with open(os.path.join(run_dir, "run_metadata.json")) as f:
        metadata = json.load(f)

    with open(os.path.join(run_dir, "bert_eval_metrics.json")) as f:
        bert_metrics = json.load(f)

    with open(os.path.join(run_dir, "bert_classification_report.txt")) as f:
        bert_report_str = f.read()

    with open(os.path.join(run_dir, "val_data.json")) as f:
        val_data = json.load(f)

    return {
        "metadata": metadata,
        "bert_metrics": bert_metrics,
        "bert_report_str": bert_report_str,
        "val_texts": val_data["texts"],
        "val_labels_encoded": val_data["labels_encoded"],
        "val_label_names_str": val_data["label_names_str"],
        "label_names": metadata["label_names"],
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


def log_comparison(
    bert_results: dict, gemini_results: dict, num_samples: int,
    dataset_name: str, bert_model_name: str, gemini_model_name: str,
) -> str:
    """Format and return a comparison summary string."""
    sep = "=" * 70
    gemini_cost_str = f"${gemini_results['cost_usd']:.4f}"
    lines = [
        sep,
        f"BERT vs {gemini_model_name} — Comparison Results",
        sep,
        f"Dataset:          {dataset_name}",
        f"BERT model:       {bert_model_name}",
        f"Gemini model:     {gemini_model_name}",
        f"Evaluation samples: {num_samples}",
        "",
        f"{'Metric':<20} {'BERT':>15} {gemini_model_name:>20}",
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
    parser = argparse.ArgumentParser(
        description="BERT vs Gemini comparison (loads BERT run, evaluates with Gemini)"
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Path to BERT run directory. If not specified, uses latest run.",
    )
    parser.add_argument(
        "--dataset", type=str, default="huffpost_news",
        help="Dataset name (used to find latest run if --run-dir not specified)",
    )
    parser.add_argument(
        "--daily-limit", type=int, default=None,
        help="Override daily Gemini request limit (default: 10000)",
    )
    args = parser.parse_args()

    # 1. Find or validate run directory
    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = find_latest_run(args.dataset)

    # 2. Setup logging
    log_path = setup_logging(dataset_name=args.dataset)
    logger.info("Comparison log: %s", log_path)
    logger.info("Using BERT run: %s", run_dir)

    # 3. Load BERT run artifacts
    bert_run = load_bert_run(run_dir)
    dataset_name = bert_run["metadata"]["dataset_name"]
    logger.info(
        "Loaded BERT run — dataset: %s, %d val samples, %d categories",
        dataset_name,
        len(bert_run["val_texts"]),
        len(bert_run["label_names"]),
    )

    # 4. Build config for Gemini settings
    config = Config(dataset_name=dataset_name)
    if args.daily_limit is not None:
        config.gemini_daily_limit = args.daily_limit

    # 5. Run Gemini with checkpoint/resume
    checkpoint_path = os.path.join(run_dir, "gemini_checkpoint.json")

    logger.info("=== Gemini: Classifying ===")
    classifier = GeminiClassifier(label_names=bert_run["label_names"])
    runner = GeminiBatchRunner(
        classifier,
        delay_seconds=config.gemini_delay_seconds,
        daily_limit=config.gemini_daily_limit,
    )

    gemini_results = runner.run(
        bert_run["val_texts"],
        bert_run["val_label_names_str"],
        checkpoint_path=checkpoint_path,
    )

    # 6. Check if complete
    if not gemini_results["completed"]:
        logger.info(
            "Daily limit reached. Processed %d/%d samples this session (%d total so far).",
            gemini_results["samples_processed_this_session"],
            len(bert_run["val_texts"]),
            gemini_results["total_samples_processed"],
        )
        logger.info("Checkpoint saved. Resume with:\n  python compare.py --run-dir %s", run_dir)
        logger.info("Full log saved to %s", log_path)
        sys.exit(0)

    # 7. Compute Gemini metrics
    preds = gemini_results["predictions"]
    trues = bert_run["val_label_names_str"]
    label_names = bert_run["label_names"]

    accuracy = accuracy_score(trues, preds)
    macro_f1 = f1_score(trues, preds, average="macro", zero_division=0)
    report = classification_report(trues, preds, labels=label_names, zero_division=0)
    logger.info("Gemini Classification Report:\n%s", report)

    gemini_model_name = GeminiClassifier.MODEL_NAME
    gemini_metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "train_time_s": 0.0,
        "eval_time_s": gemini_results["elapsed_seconds"],
        "total_time_s": gemini_results["elapsed_seconds"],
        "cost_usd": gemini_results["cost_usd"],
        "input_tokens": gemini_results["total_input_tokens"],
        "output_tokens": gemini_results["total_output_tokens"],
        "unknown_count": sum(1 for p in preds if p == "UNKNOWN"),
        "report": report,
    }

    # 8. Build BERT metrics dict for comparison (from saved artifacts)
    bert_metrics = {
        "accuracy": bert_run["bert_metrics"]["accuracy"],
        "macro_f1": bert_run["bert_metrics"]["macro_f1"],
        "train_time_s": bert_run["bert_metrics"].get("train_time_s", 0),
        "eval_time_s": bert_run["bert_metrics"].get("eval_time_s", 0),
        "total_time_s": bert_run["bert_metrics"].get("total_time_s", 0),
        "cost_usd": 0.0,
        "report": bert_run["bert_report_str"],
    }

    # 9. Log comparison
    summary = log_comparison(
        bert_metrics, gemini_metrics, len(bert_run["val_texts"]),
        dataset_name=dataset_name,
        bert_model_name=bert_run["metadata"]["model_name"],
        gemini_model_name=gemini_model_name,
    )
    logger.info("\n%s", summary)

    # 10. Save comparison file to run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = os.path.join(run_dir, f"comparison_{timestamp}.txt")
    with open(comparison_path, "w") as f:
        f.write(summary)
        f.write(f"\n\n--- BERT ({bert_run['metadata']['model_name']}) Classification Report ---\n")
        f.write(bert_metrics["report"])
        f.write(f"\n\n--- Gemini ({gemini_model_name}) Classification Report ---\n")
        f.write(gemini_metrics["report"])

    logger.info("Comparison saved to %s", comparison_path)
    logger.info("Full log saved to %s", log_path)

import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import datetime

from gemini.classifier import GeminiClassifier

logger = logging.getLogger(__name__)

# Free tier: 10 RPM. Paid tier 1: 150+ RPM.
# Adjust this based on your tier.
DEFAULT_DELAY_SECONDS = 0.5
DEFAULT_DAILY_LIMIT = 10_000


class GeminiBatchRunner:
    """
    Runs GeminiClassifier over a list of texts with rate limiting,
    checkpoint/resume support, and a configurable daily request limit.

    Tracks predictions and elapsed time for comparison with BERT.
    """

    def __init__(
        self,
        classifier: GeminiClassifier,
        delay_seconds: float = DEFAULT_DELAY_SECONDS,
        daily_limit: int = DEFAULT_DAILY_LIMIT,
    ):
        self.classifier = classifier
        self.delay_seconds = delay_seconds
        self.daily_limit = daily_limit

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_data_hash(texts: list[str]) -> str:
        """Compute a short hash of the val texts to detect stale checkpoints."""
        h = hashlib.sha256()
        h.update(str(len(texts)).encode())
        # Hash first, middle, and last texts for speed
        for idx in [0, len(texts) // 2, len(texts) - 1]:
            if idx < len(texts):
                h.update(texts[idx].encode())
        return h.hexdigest()[:16]

    @staticmethod
    def _save_checkpoint(path: str, data: dict) -> None:
        """Atomically save checkpoint to avoid corruption on crash."""
        dir_name = os.path.dirname(path)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            os.unlink(tmp_path)
            raise

    @staticmethod
    def _load_checkpoint(path: str) -> dict | None:
        """Load checkpoint if it exists, otherwise return None."""
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Main run method
    # ------------------------------------------------------------------

    def run(
        self,
        texts: list[str],
        labels: list[str],
        checkpoint_path: str | None = None,
    ) -> dict:
        """
        Classify all texts with checkpoint/resume support.

        Args:
            texts: list of news texts to classify
            labels: list of ground-truth label names (same length as texts)
            checkpoint_path: path to checkpoint JSON file. If it exists,
                resumes from where it left off. Saves progress periodically
                and when the daily limit is reached.

        Returns:
            dict with predictions, true labels, elapsed time, token counts,
            cost, and a 'completed' boolean indicating if all samples are done.
        """
        total = len(texts)
        data_hash = self._compute_data_hash(texts)

        # Try to resume from checkpoint
        start_index = 0
        predictions = []
        prior_elapsed = 0.0

        if checkpoint_path:
            checkpoint = self._load_checkpoint(checkpoint_path)
            if checkpoint is not None:
                # Verify data hash matches
                if checkpoint.get("data_hash") != data_hash:
                    logger.warning(
                        "Checkpoint data hash mismatch — val data may have changed. "
                        "Starting fresh."
                    )
                else:
                    start_index = checkpoint["next_index"]
                    predictions = checkpoint["predictions"]
                    prior_elapsed = checkpoint.get("elapsed_seconds", 0.0)

                    # Restore token counts
                    self.classifier.total_input_tokens = checkpoint.get(
                        "total_input_tokens", 0
                    )
                    self.classifier.total_output_tokens = checkpoint.get(
                        "total_output_tokens", 0
                    )

                    logger.info(
                        "Resumed from checkpoint: %d/%d samples already processed",
                        start_index, total,
                    )

        if start_index >= total:
            logger.info("All %d samples already processed — nothing to do", total)
            return self._build_result(
                predictions, labels, prior_elapsed, completed=True
            )

        logger.info(
            "Starting Gemini classification: samples %d–%d of %d "
            "(delay=%.1fs, daily_limit=%d)",
            start_index, total - 1, total,
            self.delay_seconds, self.daily_limit,
        )

        t_start = time.time()
        requests_this_session = 0

        try:
            for i in range(start_index, total):
                pred = self.classifier.classify(texts[i])
                predictions.append(pred)
                requests_this_session += 1

                if (i + 1) % 50 == 0 or (i + 1) == total:
                    elapsed = prior_elapsed + (time.time() - t_start)
                    logger.info(
                        "  Gemini progress: %d/%d (%.1fs elapsed, %d this session)",
                        i + 1, total, elapsed, requests_this_session,
                    )

                    # Save checkpoint every 50 samples
                    if checkpoint_path:
                        self._save_checkpoint(checkpoint_path, {
                            "next_index": i + 1,
                            "predictions": predictions,
                            "total_input_tokens": self.classifier.total_input_tokens,
                            "total_output_tokens": self.classifier.total_output_tokens,
                            "cost_usd": self.classifier.compute_cost(),
                            "elapsed_seconds": elapsed,
                            "data_hash": data_hash,
                            "last_updated": datetime.now().isoformat(),
                        })

                # Check daily limit
                if requests_this_session >= self.daily_limit and (i + 1) < total:
                    elapsed = prior_elapsed + (time.time() - t_start)
                    logger.info(
                        "Daily limit reached (%d requests). Saving checkpoint at sample %d/%d.",
                        self.daily_limit, i + 1, total,
                    )
                    if checkpoint_path:
                        self._save_checkpoint(checkpoint_path, {
                            "next_index": i + 1,
                            "predictions": predictions,
                            "total_input_tokens": self.classifier.total_input_tokens,
                            "total_output_tokens": self.classifier.total_output_tokens,
                            "cost_usd": self.classifier.compute_cost(),
                            "elapsed_seconds": elapsed,
                            "data_hash": data_hash,
                            "last_updated": datetime.now().isoformat(),
                        })
                    return self._build_result(
                        predictions, labels, elapsed,
                        completed=False,
                        requests_this_session=requests_this_session,
                    )

                # Rate limiting — skip delay on the last request
                if i < total - 1:
                    time.sleep(self.delay_seconds)

        except KeyboardInterrupt:
            elapsed = prior_elapsed + (time.time() - t_start)
            logger.info("Interrupted! Saving checkpoint at sample %d/%d.", len(predictions), total)
            if checkpoint_path:
                self._save_checkpoint(checkpoint_path, {
                    "next_index": len(predictions),
                    "predictions": predictions,
                    "total_input_tokens": self.classifier.total_input_tokens,
                    "total_output_tokens": self.classifier.total_output_tokens,
                    "cost_usd": self.classifier.compute_cost(),
                    "elapsed_seconds": elapsed,
                    "data_hash": data_hash,
                    "last_updated": datetime.now().isoformat(),
                })
            raise

        elapsed_total = prior_elapsed + (time.time() - t_start)

        # Final checkpoint save
        if checkpoint_path:
            self._save_checkpoint(checkpoint_path, {
                "next_index": total,
                "predictions": predictions,
                "total_input_tokens": self.classifier.total_input_tokens,
                "total_output_tokens": self.classifier.total_output_tokens,
                "cost_usd": self.classifier.compute_cost(),
                "elapsed_seconds": elapsed_total,
                "data_hash": data_hash,
                "last_updated": datetime.now().isoformat(),
            })

        correct = sum(p == t for p, t in zip(predictions, labels))
        unknown = sum(p == "UNKNOWN" for p in predictions)
        logger.info(
            "Gemini done — %d/%d correct, %d unknown, %.1fs total",
            correct, total, unknown, elapsed_total,
        )

        return self._build_result(
            predictions, labels, elapsed_total,
            completed=True,
            requests_this_session=requests_this_session,
        )

    def _build_result(
        self,
        predictions: list[str],
        labels: list[str],
        elapsed: float,
        completed: bool,
        requests_this_session: int = 0,
    ) -> dict:
        """Build the standard result dict."""
        return {
            "predictions": predictions,
            "true_labels": labels,
            "elapsed_seconds": elapsed,
            "total_input_tokens": self.classifier.total_input_tokens,
            "total_output_tokens": self.classifier.total_output_tokens,
            "cost_usd": self.classifier.compute_cost(),
            "completed": completed,
            "samples_processed_this_session": requests_this_session,
            "total_samples_processed": len(predictions),
        }

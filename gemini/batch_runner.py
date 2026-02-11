import logging
import time

from gemini.classifier import GeminiClassifier

logger = logging.getLogger(__name__)

# Free tier: 10 RPM. Paid tier 1: 150+ RPM.
# Adjust this based on your tier.
DEFAULT_DELAY_SECONDS = 1.0  # ~60 RPM, safe for paid tier 1


class GeminiBatchRunner:
    """
    Runs GeminiClassifier over a list of texts with rate limiting.

    Tracks predictions and elapsed time for comparison with BERT.
    """

    def __init__(
        self,
        classifier: GeminiClassifier,
        delay_seconds: float = DEFAULT_DELAY_SECONDS,
    ):
        self.classifier = classifier
        self.delay_seconds = delay_seconds

    def run(
        self, texts: list[str], labels: list[str]
    ) -> dict:
        """
        Classify all texts and return results.

        Args:
            texts: list of news texts to classify
            labels: list of ground-truth label names (same length as texts)

        Returns:
            dict with predictions, true labels, elapsed time, and token counts
        """
        predictions = []
        total = len(texts)

        logger.info("Starting Gemini classification of %d samples (delay=%.1fs)", total, self.delay_seconds)

        t_start = time.time()

        for i, text in enumerate(texts):
            pred = self.classifier.classify(text)
            predictions.append(pred)

            if (i + 1) % 50 == 0 or (i + 1) == total:
                elapsed = time.time() - t_start
                logger.info(
                    "  Gemini progress: %d/%d (%.1fs elapsed)", i + 1, total, elapsed
                )

            # Rate limiting — skip delay on the last request
            if i < total - 1:
                time.sleep(self.delay_seconds)

        elapsed_total = time.time() - t_start

        # Count correct predictions
        correct = sum(p == t for p, t in zip(predictions, labels))
        unknown = sum(p == "UNKNOWN" for p in predictions)

        logger.info(
            "Gemini done — %d/%d correct, %d unknown, %.1fs total",
            correct, total, unknown, elapsed_total,
        )

        return {
            "predictions": predictions,
            "true_labels": labels,
            "elapsed_seconds": elapsed_total,
            "total_input_tokens": self.classifier.total_input_tokens,
            "total_output_tokens": self.classifier.total_output_tokens,
            "cost_usd": self.classifier.compute_cost(),
        }

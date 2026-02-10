import logging
import os
from datetime import datetime
from pipeline import run_pipeline


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"


def setup_logging() -> str:
    """Configure console + file logging. Returns the log file path."""
    os.makedirs("outputs/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"outputs/logs/run_{timestamp}.txt"

    # Root logger — captures all modules that use logging.getLogger(__name__)
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(file_handler)

    return log_path


if __name__ == "__main__":
    log_path = setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Log file: %s", log_path)

    # --- Choose dataset ---
    # Option 1: BBC News (local folder)
    # results = run_pipeline(
    #     dataset_name="bbc",
    #     data_path="bbc",
    #     num_epochs=3,
    # )

    # Option 2: AG News (from HuggingFace, with sample limit for fast experimentation)
    results = run_pipeline(
        dataset_name="ag_news",
        num_epochs=3,
        max_samples=None,
    )

    logger.info("Log saved to %s", log_path)
    print(results["report_str"])

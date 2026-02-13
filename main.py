import logging
import os
from datetime import datetime
from pipeline import run_pipeline


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"


def setup_logging(dataset_name: str) -> str:
    """Configure console + file logging. Returns the log file path."""
    os.makedirs("outputs/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"outputs/logs/run_bert_{dataset_name}_{timestamp}.txt"

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
    # --- Choose dataset ---
    DATASET_NAME = "huffpost_news"

    log_path = setup_logging(dataset_name=DATASET_NAME)
    logger = logging.getLogger(__name__)
    logger.info("Log file: %s", log_path)

    # Option 1: BBC News (local folder)
    # results = run_pipeline(
    #     dataset_name="bbc",
    #     data_path="bbc",
    #     num_epochs=3,
    # )

    # Option 2: AG News (from HuggingFace, with sample limit for fast experimentation)
    # results = run_pipeline(
    #     dataset_name="ag_news",
    #     num_epochs=3,
    #     max_samples=None,
    # )

    # Option 3: HuffPost News (from HuggingFace)
    results = run_pipeline(
        dataset_name=DATASET_NAME,
        num_epochs=8,
        max_samples=None,
    )

    # Option 4: Reuters-21578 (from HuggingFace, 8 classes)
    # results = run_pipeline(
    #     dataset_name="reuters",
    #     num_epochs=3,
    #     max_samples=1000,
    # )

    logger.info("Run saved to: %s", results["run_dir"])
    logger.info("Log saved to %s", log_path)
    print(results["report_str"])

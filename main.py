import logging
from pipeline import run_pipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

if __name__ == "__main__":
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

    print(results["report_str"])

import logging

from data.bbc_data_preprocessor import BBCDataPreprocessor
from data.ag_news_preprocessor import AGNewsPreprocessor
from model.classifier import BBCBertClassifier
from test.trainer import Trainer
from test.evaluator import Evaluator
from config import Config

logger = logging.getLogger(__name__)


def run_pipeline(
    dataset_name: str = "bbc",
    data_path: str | None = None,
    num_epochs: int = 3,
    max_samples: int | None = None,
) -> dict:
    """Run the full train & evaluate pipeline. Returns evaluation results."""

    # 1. Preprocess
    config = Config(dataset_name=dataset_name, num_epochs=num_epochs, max_samples=max_samples)

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
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'bbc' or 'ag_news'.")

    train_loader, val_loader = preprocessor.run()

    # 2. Build model
    config.num_labels = preprocessor.num_labels
    model = BBCBertClassifier(config=config)

    # 3. Train
    trainer = Trainer(model, config=config)
    trainer.train(train_loader, val_loader)

    # 4. Evaluate
    config.label_names = preprocessor.label_names
    evaluator = Evaluator(model, config=config)
    results = evaluator.evaluate(val_loader)

    # 5. Save
    model.save("outputs/model")

    return results

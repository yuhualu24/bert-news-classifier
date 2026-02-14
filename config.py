from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # Dataset
    dataset_name: str = "bbc"  # "bbc" or "ag_news"
    max_samples: Optional[int] = None  # limit training/val samples (useful for large datasets)

    # Paths
    output_dir: str = "outputs"

    # Model
    model_name: str = "bert-base-uncased"
    hidden_size: int = 768
    num_labels: int = 5

    # Tokenizer
    max_length: int = 256
    padding: str = "max_length"
    truncation: bool = True

    # Data split
    test_size: float = 0.1
    random_state: int = 42

    # Training
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    early_stopping_patience: int = 2  # Stop after N epochs with no val_loss improvement

    # Gemini settings (used by compare.py)
    gemini_daily_limit: int = 10_000
    gemini_delay_seconds: float = 0.5

    # Label ordering (alphabetical by default, set after loading)
    label_names: List[str] = field(default_factory=list)
from dataclasses import dataclass

from torch.utils.data import DataLoader


@dataclass
class PreprocessorResult:
    """Standardized output from all dataset preprocessors.

    Bundles the tokenized DataLoaders (for BERT training) with the raw
    validation texts and string labels (needed for Gemini evaluation
    and saving run artifacts).
    """

    train_loader: DataLoader
    val_loader: DataLoader
    val_texts: list[str]
    val_labels_encoded: list[int]
    val_label_names_str: list[str]  # Human-readable label for each val sample
    label_names: list[str]  # All unique label names (ordered)
    num_labels: int

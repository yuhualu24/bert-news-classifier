SYSTEM_INSTRUCTION = (
    "You are a news article classifier. "
    "You will be given a news text and a list of valid category labels. "
    "Respond with ONLY the category name that best matches the text. "
    "Do not include any explanation, punctuation, or extra text."
)


def build_classification_prompt(text: str, label_names: list[str]) -> str:
    """Build a zero-shot classification prompt for a single news text."""
    labels_str = ", ".join(label_names)
    return (
        f"Classify the following news text into exactly one of these categories:\n"
        f"[{labels_str}]\n\n"
        f"Text: \"{text}\"\n\n"
        f"Category:"
    )

import logging
import os

from google import genai
from google.genai import types
from enum import Enum
from gemini.prompt import SYSTEM_INSTRUCTION, build_classification_prompt

logger = logging.getLogger(__name__)


class GeminiClassifier:
    """
    Zero-shot text classifier using Gemini 2.5 Flash.

    Sends a classification prompt for each text and parses the response
    into one of the valid category labels.
    """

    MODEL_NAME = "gemini-2.5-flash"

    # Pricing per 1M tokens (USD) — source: https://ai.google.dev/gemini-api/docs/pricing
    COST_PER_1M_INPUT_TOKENS = 0.15
    COST_PER_1M_OUTPUT_TOKENS = 0.60

    def __init__(self, label_names: list[str], api_key: str | None = None):
        self.label_names = label_names
        self._label_lookup = {name.upper().strip(): name for name in label_names}

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Gemini API key required. Pass api_key or set GEMINI_API_KEY env var."
            )

        self.client = genai.Client(api_key=resolved_key)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._category_enum = Enum(
            "Category",
            {name.replace(" ", "_").replace("&", "AND"): name for name in label_names}
        )

    def classify(self, text: str) -> str:
        """
        Classify a single text into one of the valid labels.

        Returns the matched label name, or "UNKNOWN" if the response
        doesn't match any valid label.
        """
        prompt = build_classification_prompt(text, self.label_names)

        response = self.client.models.generate_content(
            model=self.MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.0,
                max_output_tokens=200,
                response_mime_type="text/x.enum",
                response_schema=self._category_enum,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE",
                    ),
                ],
            ),
        )

        # Track token usage
        if response.usage_metadata:
            self.total_input_tokens += response.usage_metadata.prompt_token_count or 0
            self.total_output_tokens += response.usage_metadata.candidates_token_count or 0

        # Handle empty or blocked responses
        if response.text is None:
            logger.warning("Gemini returned None response")
            logger.warning(f"None response. Prompt feedback: {response.prompt_feedback}")
            if response.candidates:
                logger.warning(f"Finish reason: {response.candidates[0].finish_reason}")
            return "UNKNOWN"

        raw = response.text.strip().upper()
        matched = self._label_lookup.get(raw)

        if matched is None:
            # Fuzzy fallback: check if any label is contained in the response
            for key, name in self._label_lookup.items():
                if key in raw:
                    matched = name
                    break

        if matched is None:
            logger.warning("Unmatched Gemini response: '%s'", response.text.strip())
            return "UNKNOWN"

        return matched

    def compute_cost(self) -> float:
        """Compute total API cost in USD based on accumulated token usage."""
        input_cost = (self.total_input_tokens / 1_000_000) * self.COST_PER_1M_INPUT_TOKENS
        output_cost = (self.total_output_tokens / 1_000_000) * self.COST_PER_1M_OUTPUT_TOKENS
        return input_cost + output_cost

    def reset_token_counts(self) -> None:
        """Reset accumulated token counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0

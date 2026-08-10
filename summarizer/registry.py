"""Model metadata registry for multi-model text summarization system.

This module houses definitions and configurations for all supported
Hugging Face transformer models.
"""

from dataclasses import dataclass
from typing import Dict

@dataclass(slots=True, frozen=True)
class ModelInfo:
    """Immutable metadata container for a supported neural architecture."""
    id: str
    display_name: str
    hf_model_id: str
    max_context_tokens: int
    prefix: str

# Central Model Registry
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "bart-large": ModelInfo(
        id="bart-large",
        display_name="BART Large CNN",
        hf_model_id="facebook/bart-large-cnn",
        max_context_tokens=1024,
        prefix=""
    ),
    "t5-base": ModelInfo(
        id="t5-base",
        display_name="T5 Base",
        hf_model_id="google-t5/t5-base",
        max_context_tokens=512,
        prefix="summarize: "
    ),
    "flan-t5-base": ModelInfo(
        id="flan-t5-base",
        display_name="FLAN-T5 Base",
        hf_model_id="google/flan-t5-base",
        max_context_tokens=512,
        prefix="Summarize the following text: "
    )
}

DEFAULT_MODEL_ID = "bart-large"

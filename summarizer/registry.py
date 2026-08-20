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
    default_num_beams: int
    default_length_penalty: float
    min_length_multiplier: float

# Central Model Registry
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "bart-large": ModelInfo(
        id="bart-large",
        display_name="BART Large CNN",
        hf_model_id="facebook/bart-large-cnn",
        max_context_tokens=1024,
        prefix="",
        default_num_beams=4,
        default_length_penalty=2.0,
        min_length_multiplier=1.0
    ),
    "flan-t5-base": ModelInfo(
        id="flan-t5-base",
        display_name="FLAN-T5 Base",
        hf_model_id="google/flan-t5-base",
        max_context_tokens=512,
        prefix="Summarize the following text: ",
        default_num_beams=2,
        default_length_penalty=0.8,
        min_length_multiplier=0.4
    )
}

DEFAULT_MODEL_ID = "bart-large"

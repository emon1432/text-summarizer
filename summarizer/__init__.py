"""Summarizer Package Initialization.

Exposes clean, architectural abstraction interfaces for internal consuming controllers.
Decouples deep learning mechanics from presentation and web application logic.
"""

from summarizer.engine import SummarizationEngine, SummaryResult
from summarizer.registry import MODEL_REGISTRY, DEFAULT_MODEL_ID
from summarizer.utils import ExecutionTimer, clean_input_text, count_words, calculate_compression_rate, chunk_text_by_sentences

__all__ = [
    "SummarizationEngine",
    "SummaryResult",
    "MODEL_REGISTRY",
    "DEFAULT_MODEL_ID",
    "ExecutionTimer",
    "clean_input_text",
    "count_words",
    "calculate_compression_rate",
    "chunk_text_by_sentences",
]

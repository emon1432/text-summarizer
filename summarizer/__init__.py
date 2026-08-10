"""Summarizer Package Initialization.

Exposes clean, architectural abstraction interfaces for internal consuming controllers.
Decouples deep learning mechanics from presentation and web application logic.
"""

from summarizer.model import TransformerSummarizer, SummaryResult
from summarizer.utils import ExecutionTimer, clean_input_text, count_words, calculate_compression_rate, chunk_text_by_sentences

__all__ = [
    "TransformerSummarizer",
    "SummaryResult",
    "ExecutionTimer",
    "clean_input_text",
    "count_words",
    "calculate_compression_rate",
    "chunk_text_by_sentences",
]

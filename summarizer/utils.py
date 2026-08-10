"""Utility algorithms and mathematical helpers for text processing and metrics.

This module provides reusable enterprise utilities for sanitizing text inputs,
calculating word frequencies and data compression rates, timing model inference
durations, and performing context-aware sentence chunking for long sequences.
"""

import re
import time
import unicodedata
from typing import Optional, Any


class ExecutionTimer:
    """Context manager and helper for measuring runtime execution with millisecond precision.

    Example Usage:
        with ExecutionTimer() as timer:
            perform_long_operation()
        print(f"Elapsed: {timer.elapsed_time:.2f}s")
    """

    def __init__(self) -> None:
        """Initializes the execution timer instance with reset state."""
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._elapsed: Optional[float] = None

    def __enter__(self) -> "ExecutionTimer":
        """Starts the performance counter upon entering the context block."""
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stops the performance counter upon exiting the context block."""
        self._end_time = time.perf_counter()
        self._elapsed = self._end_time - self._start_time

    def start(self) -> None:
        """Manually starts or resets the timer execution clock."""
        self._start_time = time.perf_counter()
        self._elapsed = None

    def stop(self) -> float:
        """Manually halts the timer and computes total elapsed seconds.

        Returns:
            float: Total time elapsed between start and stop in seconds.
        """
        self._end_time = time.perf_counter()
        self._elapsed = self._end_time - self._start_time
        return self._elapsed

    @property
    def elapsed_time(self) -> float:
        """Retrieves measured elapsed execution time in seconds.

        Returns:
            float: Elapsed duration in seconds, rounded to two decimal places.
        """
        if self._elapsed is None:
            # Calculate ongoing elapsed time if not stopped yet
            current_elapsed = time.perf_counter() - self._start_time
            return round(current_elapsed, 2)
        return round(self._elapsed, 2)


def clean_input_text(text: str) -> str:
    """Sanitizes raw text input by normalizing whitespace and stripping artifacts.

    Applies Unicode NFKC normalization and filters out unprintable control characters
    while maintaining sentence formatting and punctuation.

    Args:
        text (str): Raw string provided by the client user interface.

    Returns:
        str: Cleansed and normalized textual sequence ready for model embedding.
    """
    if not text:
        return ""
    # Apply Unicode NFKC normalization to resolve invisible glyphs and non-breaking spaces
    text = unicodedata.normalize('NFKC', text)
    # Strip unprintable ASCII and Unicode control characters except standard whitespace and newlines
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Normalize carriage returns and tab characters into standard spaces
    text = text.replace('\r', ' ').replace('\t', ' ')
    # Eliminate redundant whitespace clusters while preserving standard punctuation
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def count_words(text: str) -> int:
    """Computes the precise count of semantic word tokens within a text sequence.

    Args:
        text (str): Input text string.

    Returns:
        int: Number of discrete word tokens separated by whitespace.
    """
    if not text:
        return 0
    words = text.strip().split()
    return len(words)


def calculate_compression_rate(original_words: int, summary_words: int) -> float:
    """Calculates the percentage by which the original text volume was reduced.

    Formula utilized:
        Compression Rate (%) = (1 - (summary_words / original_words)) * 100

    Args:
        original_words (int): Word token frequency of the source input text.
        summary_words (int): Word token frequency of the AI-generated summary.

    Returns:
        float: Calculated compression reduction percentage rounded to one decimal place.
               Returns 0.0 if input length is zero or invalid.
    """
    if original_words <= 0:
        return 0.0
    if summary_words >= original_words:
        return 0.0
    
    reduction_ratio = 1.0 - (float(summary_words) / float(original_words))
    compression_percent = reduction_ratio * 100.0
    return round(compression_percent, 1)


def chunk_text_by_sentences(
    text: str,
    max_words_per_chunk: int = 400,
    tokenizer: Optional[Any] = None,
    max_tokens: int = 950
) -> list[str]:
    """Divides extensive text sequences into coherent, tokenizer-safe sentence chunks.

    BART transformer models operate under an absolute attention boundary of 1024
    positional tokens. This procedure splits documents along punctuation boundaries
    and optionally verifies exact subword length via Hugging Face tokenizers to prevent
    tensor indexing exceptions on non-standard textual phrasing.

    Args:
        text (str): Cleaned full input document string.
        max_words_per_chunk (int): Threshold of word tokens allowed per individual chunk.
        tokenizer (Optional[Any]): Hugging Face tokenizer instance for precise subword checking.
        max_tokens (int): Maximum safe BPE subword tokens allowed per chunk (default 950).

    Returns:
        list[str]: An ordered list of text chunks ready for independent summarization.
    """
    if not text:
        return []

    # Verify if text already fits within word limit and (if tokenizer provided) token limit
    total_words = count_words(text)
    if total_words <= max_words_per_chunk:
        if tokenizer is None or len(tokenizer.encode(text, add_special_tokens=False)) <= max_tokens:
            return [text]

    # Split text utilizing regex on boundary delimiters (. ! ?) followed by whitespace
    # Negative lookahead protects common single-letter abbreviations or numbers
    sentence_delimiters = re.compile(r'(?<=[.!?])+\s+(?=[A-Z0-9])')
    sentences = sentence_delimiters.split(text)

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_word_count: int = 0

    for sentence in sentences:
        sentence_words = count_words(sentence)
        candidate_chunk = " ".join(current_chunk + [sentence]).strip()

        # Determine if adding this sentence surpasses word count threshold or subword token bounds
        exceeds_words = (current_word_count + sentence_words) > max_words_per_chunk
        exceeds_tokens = False
        if tokenizer is not None and len(current_chunk) > 0:
            exceeds_tokens = len(tokenizer.encode(candidate_chunk, add_special_tokens=False)) > max_tokens

        if (exceeds_words or exceeds_tokens) and current_chunk:
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = [sentence]
            current_word_count = sentence_words
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_words

    # Append residual contents remaining in active buffer
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks

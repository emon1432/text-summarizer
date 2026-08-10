"""Transformer model abstraction and PyTorch Seq2Seq summarization inference engine.

This module encapsulates the Hugging Face BART architecture within a robust
Singleton inference engine. It performs device routing (CUDA/CPU), tensor
tokenization, beam search decoding, and automatic sequence chunking for extended documents.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedModel, PreTrainedTokenizerFast

from config import Config
from summarizer.utils import (
    ExecutionTimer,
    clean_input_text,
    count_words,
    calculate_compression_rate,
    chunk_text_by_sentences
)

# Configure module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass(slots=True, frozen=True)
class SummaryResult:
    """Immutable data container holding the complete analytical output of a summarization job.

    Attributes:
        original_text (str): Sanitized source text provided by the client.
        summary_text (str): Final sequence generated via Seq2Seq decoding.
        original_word_count (int): Token frequency count of the original input.
        summary_word_count (int): Token frequency count of the summary output.
        compression_percentage (float): Mathematical volume reduction rate (%).
        processing_time_seconds (float): Precise total runtime duration in seconds.
        device_used (str): Hardware execution device leveraged during forward pass ('cuda:0' or 'cpu').
    """
    original_text: str
    summary_text: str
    original_word_count: int
    summary_word_count: int
    compression_percentage: float
    processing_time_seconds: float
    device_used: str


class TransformerSummarizer:
    """Singleton pattern implementation wrapping the PyTorch BART pre-trained architecture.

    Ensures that ~1.6 GB model weights are initialized and loaded into server memory strictly
    once per process lifecycle, preventing Out-Of-Memory (OOM) exceptions and latency degradation.
    """

    _instance: Optional["TransformerSummarizer"] = None
    _is_initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "TransformerSummarizer":
        """Controls instantiation to enforce the thread-safe Singleton design."""
        if cls._instance is None:
            logger.info("Initializing TransformerSummarizer Singleton instance...")
            cls._instance = super(TransformerSummarizer, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = Config.MODEL_NAME, device: Optional[str] = Config.DEVICE) -> None:
        """Initializes the tokenizer, weights, and compute hardware target upon first execution.

        Args:
            model_name (str): Identifier of Hugging Face transformer model checkpoint.
            device (Optional[str]): Explicit computing device target ('cuda' or 'cpu').
                                    If None, auto-detects CUDA availability via PyTorch.
        """
        if self._is_initialized:
            return

        self.model_name = model_name
        self.device = self._determine_device(device)
        logger.info(f"Target computing device selected: [{self.device}]")

        # Initialize Tokenizer and Pre-trained Seq2Seq Model
        logger.info(f"Loading pre-trained tokenizer: {self.model_name}...")
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self.model_name)
        
        logger.info(f"Loading pre-trained neural network weights into [{self.device}]...")
        self.model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Deploy model tensors onto target compute accelerator and set evaluative inference mode
        self.model.to(torch.device(self.device))
        self.model.eval()

        self.__class__._is_initialized = True
        logger.info("TransformerSummarizer neural engine initialized successfully.")

    @staticmethod
    def _determine_device(preferred_device: Optional[str]) -> str:
        """Evaluates host hardware architecture to determine optimal tensor acceleration.

        Args:
            preferred_device (Optional[str]): Configuration requested device override.

        Returns:
            str: Verified hardware device tag ('cuda:0' or 'cpu').
        """
        if preferred_device is not None:
            return preferred_device.lower()
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA accelerator detected: {gpu_name}")
            return "cuda:0"
        logger.info("No CUDA accelerators found. Optimizing for CPU multi-threaded inference.")
        return "cpu"

    def _generate_chunk_summary(
        self,
        chunk_text: str,
        max_length: int = Config.DEFAULT_MAX_LENGTH,
        min_length: int = Config.DEFAULT_MIN_LENGTH,
        num_beams: int = Config.DEFAULT_NUM_BEAMS
    ) -> str:
        """Runs a single forward pass through the transformer decoder for a specific text chunk.

        Args:
            chunk_text (str): Bounded text fragment (< 1024 positional tokens).
            max_length (int): Maximum output tokens generated via beam search.
            min_length (int): Minimum enforced tokens in generated sequence.
            num_beams (int): Beam search exploration width.

        Returns:
            str: Decoded summary string for the input chunk.
        """
        # Tokenize text and route tensors to target hardware
        inputs = self.tokenizer(
            chunk_text,
            max_length=Config.TOKEN_CHUNK_LIMIT,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Disable gradient calculations during inference to conserve RAM and enhance speed
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=Config.LENGTH_PENALTY,
                early_stopping=True
            )

        # Decode generated token IDs back into readable string representations
        summary_text = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return summary_text.strip()

    def summarize(
        self,
        raw_text: str,
        max_length: int = Config.DEFAULT_MAX_LENGTH,
        min_length: int = Config.DEFAULT_MIN_LENGTH,
        num_beams: int = Config.DEFAULT_NUM_BEAMS
    ) -> SummaryResult:
        """Orchestrates end-to-end processing, chunking, AI inference, and metric analytics.

        Args:
            raw_text (str): Unprocessed string supplied by user interface.
            max_length (int): Maximum output tokens permitted per summarized sequence.
            min_length (int): Minimum required sequence token ceiling.
            num_beams (int): Number of concurrent search paths evaluated during generation.

        Returns:
            SummaryResult: Immutable data structure bundling input, output, and KPIs.
        """
        logger.info("Starting text summarization job execution...")
        
        # Enforce hyperparameter boundaries for model stability
        max_len = max(20, min(500, int(max_length)))
        min_len = max(5, min(300, int(min_length)))
        beams = max(1, min(8, int(num_beams)))
        if min_len >= max_len:
            min_len = max(5, max_len - 10)

        with ExecutionTimer() as timer:
            # 1. Clean and normalize source document text
            cleaned_text = clean_input_text(raw_text)
            orig_word_count = count_words(cleaned_text)

            if orig_word_count == 0:
                logger.warning("Empty text input provided. Bypassing neural execution.")
                return SummaryResult(
                    original_text=raw_text,
                    summary_text="No input text provided for summarization.",
                    original_word_count=0,
                    summary_word_count=0,
                    compression_percentage=0.0,
                    processing_time_seconds=0.0,
                    device_used=self.device
                )

            # 2. Divide extended prose into tokenizer-safe contextual chunks (< 950 BPE tokens)
            chunks = chunk_text_by_sentences(
                cleaned_text,
                max_words_per_chunk=400,
                tokenizer=self.tokenizer,
                max_tokens=950
            )
            logger.info(f"Input document segmented into {len(chunks)} independent execution chunk(s).")

            # 3. Perform autoregressive transformer Seq2Seq inference across all chunks
            summarized_chunks: list[str] = []
            for idx, chunk in enumerate(chunks, 1):
                logger.debug(f"Executing sequence generation on chunk {idx}/{len(chunks)}...")
                # Dynamically adjust min/max lengths for smaller trailing chunks if needed
                chunk_words = count_words(chunk)
                effective_max_len = max(15, min(max_len, max(30, chunk_words)))
                # Enforce that effective_min_len is strictly below effective_max_len to prevent PyTorch generation exceptions
                effective_min_len = min(min_len, int(effective_max_len * 0.4))
                if effective_min_len >= effective_max_len:
                    effective_min_len = max(1, effective_max_len - 5)

                chunk_summary = self._generate_chunk_summary(
                    chunk_text=chunk,
                    max_length=effective_max_len,
                    min_length=effective_min_len,
                    num_beams=beams
                )
                summarized_chunks.append(chunk_summary)

            # 4. Merge generated sequences into a unified summary and calculate analytics
            final_summary_text = " ".join(summarized_chunks).strip()
            summary_word_count = count_words(final_summary_text)
            compression_rate = calculate_compression_rate(orig_word_count, summary_word_count)

        logger.info(f"Summarization complete! Elapsed time: {timer.elapsed_time}s | Compression: {compression_rate}%")
        
        return SummaryResult(
            original_text=cleaned_text,
            summary_text=final_summary_text,
            original_word_count=orig_word_count,
            summary_word_count=summary_word_count,
            compression_percentage=compression_rate,
            processing_time_seconds=timer.elapsed_time,
            device_used=self.device
        )

"""Summarization engine orchestrator and memory manager.

This module encapsulates the multi-model architecture, managing the active
model adapter and performing automatic memory unloading to prevent OOM errors.
"""

import logging
import gc
from dataclasses import dataclass
from typing import Optional, Any
import torch

from config import Config
from summarizer.registry import MODEL_REGISTRY, DEFAULT_MODEL_ID
from summarizer.adapters import BaseModelAdapter, BartAdapter, T5Adapter
from summarizer.utils import (
    ExecutionTimer,
    clean_input_text,
    count_words,
    calculate_compression_rate,
    chunk_text_by_sentences
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass(slots=True, frozen=True)
class SummaryResult:
    """Immutable data container holding the complete analytical output of a summarization job."""
    original_text: str
    summary_text: str
    original_word_count: int
    summary_word_count: int
    compression_percentage: float
    processing_time_seconds: float
    device_used: str
    model_display_name: str


class SummarizationEngine:
    """Singleton engine orchestrating multiple text summarization models.

    Ensures only one model is loaded into RAM/VRAM at any given time.
    Safely unloads the previous model when a different model is requested.
    """

    _instance: Optional["SummarizationEngine"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "SummarizationEngine":
        if cls._instance is None:
            logger.info("Initializing SummarizationEngine Singleton instance...")
            cls._instance = super(SummarizationEngine, cls).__new__(cls)
            cls._instance._is_initialized = False
        return cls._instance

    def __init__(self, device: Optional[str] = Config.DEVICE) -> None:
        if self._is_initialized:
            return

        self.device = self._determine_device(device)
        self.active_adapter: Optional[BaseModelAdapter] = None
        self.active_model_id: Optional[str] = None
        
        logger.info(f"Target computing device selected: [{self.device}]")
        self._is_initialized = True

    @staticmethod
    def _determine_device(preferred_device: Optional[str]) -> str:
        if preferred_device is not None:
            return preferred_device.lower()
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA accelerator detected: {gpu_name}")
            return "cuda:0"
        logger.info("No CUDA accelerators found. Optimizing for CPU multi-threaded inference.")
        return "cpu"

    def _load_model(self, model_id: str) -> None:
        """Loads the requested model, replacing the current one if necessary."""
        if model_id not in MODEL_REGISTRY:
            logger.warning(f"Model ID '{model_id}' not found in registry. Falling back to default.")
            model_id = DEFAULT_MODEL_ID
            
        if self.active_model_id == model_id and self.active_adapter is not None:
            # Model is already loaded
            return
            
        # Unload existing model if present
        if self.active_adapter is not None:
            logger.info(f"Unloading previous model: {self.active_model_id}")
            self.active_adapter.unload()
            del self.active_adapter
            self.active_adapter = None
            
            # Force garbage collection and CUDA cache clear to prevent OOM
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Load new model
        logger.info(f"Loading new model architecture: {model_id}...")
        model_info = MODEL_REGISTRY[model_id]
        
        # Instantiate the correct adapter
        if model_id.startswith("bart"):
            self.active_adapter = BartAdapter(model_info, self.device)
        elif model_id.startswith("t5") or model_id.startswith("flan-t5"):
            self.active_adapter = T5Adapter(model_info, self.device)
        else:
            raise ValueError(f"No adapter implemented for model ID: {model_id}")
            
        self.active_adapter.load()
        self.active_model_id = model_id
        logger.info(f"Successfully loaded {model_info.display_name} into [{self.device}].")

    def summarize(
        self,
        raw_text: str,
        max_length: int = Config.DEFAULT_MAX_LENGTH,
        min_length: int = Config.DEFAULT_MIN_LENGTH,
        model_id: str = DEFAULT_MODEL_ID
    ) -> SummaryResult:
        """Orchestrates chunking and generation for the specified model."""
        logger.info(f"Starting text summarization job using model: {model_id}")
        
        # Ensure correct model is loaded in memory
        self._load_model(model_id)
        adapter = self.active_adapter
        
        max_len = max(20, min(500, int(max_length)))
        base_min_len = int(min_length * adapter.model_info.min_length_multiplier)
        min_len = max(5, min(300, base_min_len))
        beams = max(1, min(8, adapter.model_info.default_num_beams))
        if min_len >= max_len:
            min_len = max(5, max_len - 10)

        with ExecutionTimer() as timer:
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
                    device_used=self.device,
                    model_display_name=adapter.model_info.display_name
                )

            # Phase 4 update: Use adapter-provided token limits
            max_context = adapter.model_info.max_context_tokens
            safe_bpe_limit = max_context - 50  # Leave room for special tokens and prefix
            
            # Segment source document into initial chunks
            current_chunks = chunk_text_by_sentences(
                cleaned_text,
                max_words_per_chunk=350, # Slightly lower to accommodate T5's smaller 512 window
                tokenizer=adapter.tokenizer,
                max_tokens=safe_bpe_limit
            )
            logger.info(f"Input document segmented into {len(current_chunks)} independent execution chunk(s) (Limit: {safe_bpe_limit} tokens).")

            stage = 1
            while len(current_chunks) > 1 or stage == 1:
                # For recursive stages, combine previous summaries and re-chunk if needed
                if stage > 1:
                    combined_text = " ".join(current_chunks).strip()
                    combined_tokens = len(adapter.tokenizer.encode(combined_text, add_special_tokens=False))
                    
                    if combined_tokens <= safe_bpe_limit:
                        current_chunks = [combined_text]
                    else:
                        new_chunks = chunk_text_by_sentences(
                            combined_text,
                            max_words_per_chunk=350,
                            tokenizer=adapter.tokenizer,
                            max_tokens=safe_bpe_limit
                        )
                        # Safety circuit breaker to prevent infinite loops if compression stalls
                        if len(new_chunks) >= len(current_chunks):
                            logger.warning("Compression plateau reached in hierarchical summarization. Stopping early.")
                            break
                        current_chunks = new_chunks

                if len(current_chunks) == 1 and stage > 1:
                    logger.info(f"Executing final global hierarchical pass (Stage {stage})...")
                elif stage > 1:
                    logger.info(f"Executing intermediate hierarchical pass (Stage {stage}) with {len(current_chunks)} chunks...")

                summarized_chunks: list[str] = []
                for idx, chunk in enumerate(current_chunks, 1):
                    logger.debug(f"Executing sequence generation on chunk {idx}/{len(current_chunks)}...")
                    chunk_words = count_words(chunk)
                    effective_max_len = max(15, min(max_len, max(30, chunk_words)))
                    effective_min_len = min(min_len, int(effective_max_len * 0.4))
                    if effective_min_len >= effective_max_len:
                        effective_min_len = max(1, effective_max_len - 5)

                    chunk_summary = adapter.generate_chunk(
                        chunk_text=chunk,
                        max_length=effective_max_len,
                        min_length=effective_min_len,
                        num_beams=beams
                    )
                    summarized_chunks.append(chunk_summary)

                current_chunks = summarized_chunks
                stage += 1
                
                # If we successfully reduced down to 1 chunk and processed it, we are done
                if len(current_chunks) == 1:
                    break

            final_summary_text = " ".join(current_chunks).strip()
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
            device_used=self.device,
            model_display_name=adapter.model_info.display_name
        )

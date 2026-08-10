"""Adapter layer standardizing diverse Transformer model interfaces.

Provides an abstract Base class and concrete implementations to normalize
quirks (like prefixes) across models such as BART and T5.
"""

import abc
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedModel, PreTrainedTokenizerFast
from summarizer.registry import ModelInfo
from config import Config

class BaseModelAdapter(abc.ABC):
    """Abstract interface defining required behaviors for summarization adapters."""

    def __init__(self, model_info: ModelInfo, device: str) -> None:
        self.model_info = model_info
        self.device = device
        
        # We assign model and tokenizer as None initially for explicit lifecycle management
        self.model: PreTrainedModel = None
        self.tokenizer: PreTrainedTokenizerFast = None

    def load(self) -> None:
        """Loads pre-trained weights and tokenizer into the specified hardware device."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_info.hf_model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_info.hf_model_id)
        self.model.to(torch.device(self.device))
        self.model.eval()

    def unload(self) -> None:
        """Purges model from memory to prevent Out-Of-Memory exceptions."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        self.model = None
        self.tokenizer = None

    @abc.abstractmethod
    def generate_chunk(
        self,
        chunk_text: str,
        max_length: int,
        min_length: int,
        num_beams: int
    ) -> str:
        """Executes a single forward pass inference on a text chunk."""
        pass


class BartAdapter(BaseModelAdapter):
    """Concrete adapter for facebook/bart architectures."""

    def generate_chunk(
        self,
        chunk_text: str,
        max_length: int,
        min_length: int,
        num_beams: int
    ) -> str:
        # BART requires no prefix
        inputs = self.tokenizer(
            chunk_text,
            max_length=self.model_info.max_context_tokens,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

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

        return self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()


class T5Adapter(BaseModelAdapter):
    """Concrete adapter for T5 and FLAN-T5 architectures."""

    def generate_chunk(
        self,
        chunk_text: str,
        max_length: int,
        min_length: int,
        num_beams: int
    ) -> str:
        # T5 family requires a task-specific prompt prefix
        input_text = f"{self.model_info.prefix}{chunk_text}"
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.model_info.max_context_tokens,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

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

        return self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

import logging
import sys
from transformers import AutoTokenizer
from summarizer.engine import SummarizationEngine
from summarizer.utils import chunk_text_by_sentences, count_words
from summarizer.registry import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO)

# Repeated string to make 670+ words
text = """The results of the double-blind placebo-controlled study indicate a statistically significant reduction in baseline cortisol levels among participants who engaged in 30 minutes of daily mindfulness meditation over an eight-week period (p < 0.01). Furthermore, self-reported anxiety metrics, as measured by the GAD-7 scale, demonstrated a corresponding decline. These findings suggest that structured mindfulness interventions may serve as a viable non-pharmacological adjunct therapy for generalized anxiety disorders. """ * 15

def debug():
    engine = SummarizationEngine(device="cpu")
    models = ["bart-large", "flan-t5-base"]
    
    with open("chunk_debug_log.txt", "w", encoding="utf-8") as f:
        f.write(f"Total input words: {count_words(text)}\n\n")
        
        for m in models:
            f.write(f"=== MODEL: {m} ===\n")
            engine._load_model(m)
            adapter = engine.active_adapter
            max_context = adapter.model_info.max_context_tokens
            safe_bpe_limit = max_context - 50
            
            chunks = chunk_text_by_sentences(
                text,
                max_words_per_chunk=350,
                tokenizer=adapter.tokenizer,
                max_tokens=safe_bpe_limit
            )
            
            f.write(f"Number of chunks: {len(chunks)}\n\n")
            
            for i, chunk in enumerate(chunks, 1):
                chunk_words = count_words(chunk)
                chunk_tokens = len(adapter.tokenizer.encode(chunk, add_special_tokens=False))
                
                f.write(f"Chunk {i}:\n")
                f.write(f"  Word count: {chunk_words}\n")
                f.write(f"  Token count: {chunk_tokens}\n")
                f.write(f"  First 100 chars: {chunk[:100]}...\n")
                f.write(f"  Last 100 chars: ...{chunk[-100:]}\n")
                
                effective_max_len = max(15, min(150, max(30, chunk_words)))
                base_min_len = int(45 * adapter.model_info.min_length_multiplier)
                effective_min_len = min(max(5, min(300, base_min_len)), int(effective_max_len * 0.4))
                if effective_min_len >= effective_max_len:
                    effective_min_len = max(1, effective_max_len - 5)
                
                summary = adapter.generate_chunk(
                    chunk, 
                    max_length=effective_max_len, 
                    min_length=effective_min_len, 
                    num_beams=adapter.model_info.default_num_beams
                )
                
                f.write(f"  summary: {summary}\n\n")
            
if __name__ == "__main__":
    debug()

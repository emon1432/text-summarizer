import logging
import sys
from summarizer.engine import SummarizationEngine
from summarizer.registry import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FinalTest")

text = """The results of the double-blind placebo-controlled study indicate a statistically significant reduction in baseline cortisol levels among participants who engaged in 30 minutes of daily mindfulness meditation over an eight-week period (p < 0.01). Furthermore, self-reported anxiety metrics, as measured by the GAD-7 scale, demonstrated a corresponding decline. These findings suggest that structured mindfulness interventions may serve as a viable non-pharmacological adjunct therapy for generalized anxiety disorders. """ * 10

def test():
    engine = SummarizationEngine(device="cpu")
    models = ["bart-large", "flan-t5-base", "bart-large"]
    
    for m in models:
        logger.info(f"\n--- Testing {m} ---")
        res = engine.summarize(text, max_length=150, min_length=45, model_id=m)
        logger.info(f"Model: {res.model_display_name}")
        logger.info(f"Words: {res.summary_word_count}/{res.original_word_count} ({res.compression_percentage}%)")
        logger.info(f"Time: {res.processing_time_seconds}s")
        logger.info(f"Summary: {res.summary_text}")
        
        assert res.summary_word_count > 0, "Summary is empty!"
        assert res.model_display_name == MODEL_REGISTRY[m].display_name

if __name__ == "__main__":
    test()

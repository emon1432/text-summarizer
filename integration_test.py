import os
import sys
import logging
import traceback
from config import Config
from summarizer.engine import SummarizationEngine
from summarizer.registry import MODEL_REGISTRY

# Setup logging to console
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger("IntegrationTest")

short_text = "The quick brown fox jumps over the lazy dog. It was a sunny day in the forest."
medium_text = """The James Webb Space Telescope is the premier space science observatory of the next decade. 
It will solve mysteries in our solar system, look beyond to distant worlds around other stars, and probe the mysterious structures and origins of our universe and our place in it. 
Webb is an international program led by NASA with its partners, ESA (European Space Agency) and the Canadian Space Agency. 
It launched on December 25, 2021, and has since been delivering unprecedented images of deep space."""

# Force long text for chunking (roughly 1200 words)
long_text = " ".join([medium_text] * 20) 

test_cases = [
    # Individual basic loading and generation
    ("bart-large", short_text, "BART Large - Short text"),
    ("t5-base", short_text, "T5 Base - Short text"),
    ("flan-t5-base", short_text, "FLAN-T5 Base - Short text"),
    
    # Chunking and longer inputs
    ("t5-base", medium_text, "T5 Base - Medium text"),
    ("flan-t5-base", long_text, "FLAN-T5 Base - Long text (Chunking test)"),
    ("bart-large", long_text, "BART Large - Long text (Chunking test)"),
    
    # Model switching stress tests
    ("t5-base", short_text, "Switch to T5 Base"),
    ("flan-t5-base", short_text, "Switch to FLAN-T5 Base"),
    ("bart-large", short_text, "Switch back to BART Large"),
]

def run_tests():
    engine = SummarizationEngine(device="cpu")
    
    passed_count = 0
    failed_count = 0
    
    for model_id, text, desc in test_cases:
        logger.info(f"--- Running Test: {desc} [{model_id}] ---")
        try:
            res = engine.summarize(
                raw_text=text,
                max_length=60,
                min_length=15,
                num_beams=2,
                model_id=model_id
            )
            
            # Validation assertions
            assert res.summary_text and len(res.summary_text.strip()) > 0, "Summary text is empty."
            assert res.original_word_count > 0, "Original word count is 0."
            assert res.summary_word_count > 0, "Summary word count is 0."
            assert res.processing_time_seconds > 0, "Processing time <= 0."
            assert res.model_display_name == MODEL_REGISTRY[model_id].display_name, "Model display name mismatch."
            assert res.device_used == "cpu", f"Expected CPU but got {res.device_used}"
            
            logger.info(f"✅ PASS: {res.summary_word_count}/{res.original_word_count} words ({res.compression_percentage}%) | {res.processing_time_seconds}s")
            passed_count += 1
            
        except Exception as e:
            logger.error(f"❌ FAIL: {e}")
            traceback.print_exc()
            failed_count += 1
            
    logger.info(f"--- TEST SUMMARY: {passed_count} PASSED, {failed_count} FAILED ---")

if __name__ == "__main__":
    run_tests()

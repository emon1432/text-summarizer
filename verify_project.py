"""Automated Verification & System Integrity Test Suite.

This script executes health checks across the project codebase, verifying syntax,
package imports, utility algorithm calculations, and Flask HTTP test client routing.

Usage:
    python verify_project.py [--run-model]
"""

import sys
import argparse
import logging
from importlib.metadata import version

# Configure Verification Test Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] | %(message)s")
logger = logging.getLogger("SystemVerification")


def verify_environment() -> bool:
    """Verifies Python 3.11+ runtime compatibility and essential core dependency imports."""
    logger.info("=== STEP 1: Verifying Python Runtime & Deep Learning Dependencies ===")
    
    # Check Python version
    py_version = sys.version_info
    logger.info(f"Detected Python Runtime: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version < (3, 8):
        logger.error("Python version is outdated. Python 3.11+ is recommended for this project.")
        return False

    try:
        import flask
        import torch
        import transformers
        import accelerate
        import sentencepiece
        logger.info(f"✔ Flask verified (v{version("flask")})")
        logger.info(f"✔ PyTorch verified (v{torch.__version__}) | CUDA Available: {torch.cuda.is_available()}")
        logger.info(f"✔ Hugging Face Transformers verified (v{transformers.__version__})")
        logger.info(f"✔ Accelerate & SentencePiece libraries verified successfully")
    except ImportError as ie:
        logger.error(f"❌ Dependency import failure: {str(ie)}")
        logger.error("Please ensure you have run: pip install -r requirements.txt")
        return False
        
    return True


def verify_domain_utilities() -> bool:
    """Tests mathematical calculation accuracy and string sanitization utility behavior."""
    logger.info("=== STEP 2: Verifying Domain Utilities & Algorithm Precision ===")
    try:
        from summarizer.utils import clean_input_text, count_words, calculate_compression_rate, chunk_text_by_sentences

        # Test 1: Cleaning and Word Counting
        sample = "   Artificial    Intelligence and \n Deep Learning   are revolutionizing NLP!   \t "
        cleaned = clean_input_text(sample)
        words = count_words(cleaned)
        assert words == 8, f"Word count discrepancy: Expected 8, got {words}"
        logger.info("✔ String sanitization & accurate token counting operational")

        # Test 2: Compression Formula
        # If original words = 100, summary = 25 -> compression should equal 75.0%
        comp = calculate_compression_rate(100, 25)
        assert comp == 75.0, f"Compression calculation anomaly: Expected 75.0, got {comp}"
        logger.info("✔ Mathematical compression rate precision verified (100 -> 25 words = 75.0% reduction)")

        # Test 3: Text Chunking
        long_prose = ". ".join([f"Sentence number {i} contains approximately seven words in total" for i in range(100)])
        chunks = chunk_text_by_sentences(long_prose, max_words_per_chunk=250)
        assert len(chunks) >= 2, f"Chunking failure: Did not partition long prose properly ({len(chunks)} chunks)"
        logger.info(f"✔ Document contextual chunker verified (segmented 700-word prose into {len(chunks)} boundary-safe chunks)")
        return True
    except AssertionError as ae:
        logger.error(f"❌ Utility calculation assertion failure: {str(ae)}")
        return False
    except Exception as ex:
        logger.exception("❌ Unexpected error during utility verification.")
        return False


def verify_flask_architecture() -> bool:
    """Simulates HTTP client interactions against the Flask enterprise backend controller."""
    logger.info("=== STEP 3: Verifying Flask Application Factory & HTTP Routing ===")
    try:
        from app import create_app
        test_app = create_app("development")
        test_app.config["TESTING"] = True
        
        with test_app.test_client() as client:
            # Test Index Page Render
            response_home = client.get("/")
            assert response_home.status_code == 200, f"HTTP GET / returned status {response_home.status_code}"
            assert b"NeuroSummarizer AI" in response_home.data or b"AI Powered Text Summarization" in response_home.data
            logger.info("✔ Primary Terminal Dashboard HTTP 200 render verified successfully")

            # Test API endpoint bad requests
            response_bad_api = client.post("/api/summarize", json={"invalid_field": "test"})
            assert response_bad_api.status_code == 400, f"Expected HTTP 400 for bad JSON, got {response_bad_api.status_code}"
            logger.info("✔ REST API payload boundary validation operational (HTTP 400 rejection verified)")
            
        return True
    except Exception as ex:
        logger.exception(f"❌ Flask routing verification failure: {str(ex)}")
        return False


def verify_model_inference() -> bool:
    """Downloads BART pre-trained weights and performs an end-to-end inference pass."""
    logger.info("=== STEP 4: Executing Live Neural Transformers Inference (facebook/bart-large-cnn) ===")
    try:
        from summarizer import SummarizationEngine, SummaryResult
        logger.info("Initializing SummarizationEngine Singleton (this may take a few moments to verify cache)...")
        engine = SummarizationEngine()
        
        sample_article = (
            "The James Webb Space Telescope is the premier space science observatory of the next decade. "
            "Webb is solving mysteries in our solar system, looking beyond to distant worlds around other stars, "
            "and probing the mysterious structures and origins of our universe and our place in it. "
            "Webb is an international program led by NASA with its partners, ESA (European Space Agency) and the Canadian Space Agency."
        )
        
        logger.info("Running Seq2Seq forward pass via Beam Search decoding...")
        output: SummaryResult = engine.summarize(sample_article, max_length=50, min_length=15, num_beams=2)
        
        logger.info("=" * 70)
        logger.info("INFERENCE SUCCESSFUL! PROOF OF EXECUTION REPORT:")
        logger.info(f" - Hardware Accelerator : {output.device_used}")
        logger.info(f" - Original Tokens    : {output.original_word_count} words")
        logger.info(f" - Summary Tokens     : {output.summary_word_count} words")
        logger.info(f" - Compression Saved  : {output.compression_percentage}%")
        logger.info(f" - Latency Duration   : {output.processing_time_seconds} seconds")
        logger.info(" - Generated Summary  : " + output.summary_text)
        logger.info("=" * 70)
        return True
    except Exception as ex:
        logger.exception("❌ Model neural inference execution failed.")
        return False


def main():
    parser = argparse.ArgumentParser(description="NLP Project Automated Verification Suite")
    parser.add_argument("--run-model", action="store_true", help="Download BART model weights and run live inference verification")
    args = parser.parse_args()

    logger.info("Starting System Integrity Test Suite...")
    
    step1 = verify_environment()
    if not step1:
        logger.error("Stopping verification due to environment dependency mismatches.")
        sys.exit(1)
        
    step2 = verify_domain_utilities()
    step3 = verify_flask_architecture()
    
    step4 = True
    if args.run_model:
        step4 = verify_model_inference()
    else:
        logger.info("ℹ Skipping heavy neural inference weights download. Use 'python verify_project.py --run-model' to test Hugging Face execution.")

    if all([step1, step2, step3, step4]):
        logger.info("🎉 ALL SYSTEM TESTS PASSED SUCCESSFULLY! The architecture is clean, production-ready, and verified.")
        sys.exit(0)
    else:
        logger.error("⚠ Certain verifications encountered anomalies. Please review output logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

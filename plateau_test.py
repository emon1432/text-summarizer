import logging
from summarizer.engine import SummarizationEngine

logging.basicConfig(level=logging.INFO)

class MockAdapter:
    def __init__(self, engine):
        self.tokenizer = engine.active_adapter.tokenizer
        self.model_info = engine.active_adapter.model_info
        self.call_count = 0
        
    def generate_chunk(self, chunk_text, **kwargs):
        self.call_count += 1
        # Always return the same length to force a plateau
        return chunk_text
        
def test_hierarchical():
    engine = SummarizationEngine(device="cpu")
    engine._load_model("bart-large")
    
    mock = MockAdapter(engine)
    engine.active_adapter.generate_chunk = mock.generate_chunk
    
    text = "This is a sentence. " * 500 # 1500 words, chunks by sentence.
    
    res = engine.summarize(text)
    print("SUCCESS!")
    print(f"Total generate calls: {mock.call_count}")
    print(f"Final summary words: {len(res.summary_text.split())}")
    
if __name__ == "__main__":
    test_hierarchical()

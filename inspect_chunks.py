import sys
from transformers import AutoTokenizer
from summarizer.utils import chunk_text_by_sentences, count_words

text = """The results of the double-blind placebo-controlled study indicate a statistically significant reduction in baseline cortisol levels among participants who engaged in 30 minutes of daily mindfulness meditation over an eight-week period (p < 0.01). Furthermore, self-reported anxiety metrics, as measured by the GAD-7 scale, demonstrated a corresponding decline. These findings suggest that structured mindfulness interventions may serve as a viable non-pharmacological adjunct therapy for generalized anxiety disorders. """ * 10

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

chunks = chunk_text_by_sentences(text, max_words_per_chunk=350, tokenizer=tokenizer, max_tokens=974)

print("Total input words:", count_words(text))
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks, 1):
    words = count_words(chunk)
    tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
    print(f"\nChunk {i}:")
    print(f"  Word count: {words}")
    print(f"  Token count: {tokens}")
    print(f"  First 100 chars: {repr(chunk[:100])}")
    print(f"  Last 100 chars: {repr(chunk[-100:])}")

if len(chunks) >= 2:
    print(f"\nChunk 1 == Chunk 2? {chunks[0] == chunks[1]}")
    # Compare semantically
    print(f"Chunk 1 length: {len(chunks[0])}, Chunk 2 length: {len(chunks[1])}")

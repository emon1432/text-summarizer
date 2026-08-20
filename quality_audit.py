import os
import sys
import logging
from summarizer.engine import SummarizationEngine
from summarizer.registry import MODEL_REGISTRY

# Setup logging
logging.basicConfig(level=logging.ERROR)

texts = {
    "1_News": """A major storm system sweeping across the Midwest has caused widespread flight cancellations at O'Hare International Airport. Thousands of travelers were stranded on Tuesday as snow and freezing rain disrupted holiday travel. Local authorities have declared a state of emergency and are urging residents to stay off the roads. Utility companies report over 50,000 homes are without power. Weather conditions are expected to improve by Thursday morning.""",
    
    "2_Academic": """The results of the double-blind placebo-controlled study indicate a statistically significant reduction in baseline cortisol levels among participants who engaged in 30 minutes of daily mindfulness meditation over an eight-week period (p < 0.01). Furthermore, self-reported anxiety metrics, as measured by the GAD-7 scale, demonstrated a corresponding decline. These findings suggest that structured mindfulness interventions may serve as a viable non-pharmacological adjunct therapy for generalized anxiety disorders.""",
    
    "3_Technical": """To deploy the application to the Kubernetes cluster, first ensure that your kubectl context is pointed to the production environment. You must build the Docker image using the provided Dockerfile and push it to the remote container registry. Once the image is available, update the deployment.yaml file to reference the new image tag. Apply the configuration using 'kubectl apply -f deployment.yaml'. Finally, monitor the rollout status with 'kubectl rollout status deployment/webapp'. If the health checks fail, the system will automatically rollback to the previous stable revision.""",
    
    "4_Prose": """The old clockmaker lived alone at the edge of the village. His shop was filled with thousands of ticking timepieces, a chaotic symphony of gears and springs. Every morning, he would carefully wind each one, a ritual that brought him an odd sense of comfort. He had spent his entire life trying to craft the perfect clock, one that would never lose a single second. Yet, as he watched his own hands tremble while adjusting a delicate mainspring, he realized that time was the one thing he could never truly master.""",
    
    "5_LongText": """Artificial intelligence and deep natural language processing have undergone transformative paradigm shifts with the advent of attention-based sequence-to-sequence transformer models. Historically, Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) architectures suffered from gradient vanishing challenges when tasked with evaluating lengthy documents, causing significant context degradation across sequential timestamps. To overcome these structural bottlenecks, bidirectional transformers such as BERT revolutionized contextual representation modeling through self-attention mechanics and masked language modeling objectives. Simultaneously, autoregressive models like OpenAI's GPT introduced exceptional generative text capabilities by predicting sequential tokens in a left-to-right decoding framework. In response to the distinct structural advantages of both encoding and decoding methodologies, researchers at Meta AI formulated BART (Bidirectional and Auto-Regressive Transformers). BART conceptually unifies a bidirectional BERT encoder with an autoregressive GPT decoder. During pre-training, input documents are purposely degraded by applying arbitrary noising transformations—including sentence shuffling, text intrenching, and continuous span masking. The neural decoder is subsequently tasked with reconstructing the original pristine text via cross-attention mechanisms over the encoder's representations. When specifically fine-tuned on comprehensive journalistic repositories such as the CNN/DailyMail abstractive summarization dataset, BART demonstrates remarkable cognitive abstraction. Unlike extractive summarizers that simply rank and extract isolated verbatim sentences, abstractive transformer architectures actively synthesize information, condense complex clauses, and construct entirely new grammatical compositions that faithfully preserve fundamental semantic propositions while drastically diminishing overall data volume.""" * 3 # Replicated to force chunking
}

models = ["bart-large", "flan-t5-base"]

def main():
    engine = SummarizationEngine(device="cpu")
    
    with open("audit_results.txt", "w", encoding="utf-8") as f:
        for text_type, content in texts.items():
            f.write(f"\n{'='*50}\n")
            f.write(f"TEXT TYPE: {text_type}\n")
            f.write(f"LENGTH: {len(content.split())} words\n")
            f.write(f"{'='*50}\n\n")
            
            for model in models:
                f.write(f"--- MODEL: {model} ---\n")
                try:
                    res = engine.summarize(
                        raw_text=content,
                        max_length=60,
                        min_length=15,
                        model_id=model
                    )
                    f.write(f"Summary ({res.summary_word_count} words):\n{res.summary_text}\n\n")
                except Exception as e:
                    f.write(f"ERROR: {str(e)}\n\n")

if __name__ == "__main__":
    main()

# AI Powered Text Summarization using Pre-trained Transformer Models

**An Application in Artificial Intelligence & Machine Learning**

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework: Flask](https://img.shields.io/badge/framework-Flask%203.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![Model: BART-Large-CNN](https://img.shields.io/badge/model-facebook%2Fbart--large--cnn-purple.svg)](https://huggingface.co/facebook/bart-large-cnn)
[![Library: PyTorch & Transformers](https://img.shields.io/badge/AI-PyTorch%20%7C%20Transformers-orange.svg)](https://huggingface.co/docs/transformers/index)

---

## 📌 Executive Summary
**AI Powered Text Summarization** is an advanced, clean-architecture web platform designed to extract concise, accurate, and grammatically articulate summaries from lengthy English prose and technical documents. Powered by Meta AI's **BART (Bidirectional and Auto-Regressive Transformers)** architecture fine-tuned on the CNN/DailyMail dataset (`facebook/bart-large-cnn`), this application integrates deep neural network sequence-to-sequence generation into a high-performance Flask enterprise backend coupled with a responsive Bootstrap 5 visual interface.

---

## 🏛️ System Architecture & Clean Domain Separation
The software engineering design strictly adheres to **Clean Architecture** patterns, modular decoupling, and **PEP8** quality standards, preventing domain logic entanglement within presentation routing layers.

```
text-summarizer/
│
├── app.py                     # HTTP Enterprise Controller & Flask routing handlers
├── config.py                  # Environment-aware hyperparameter & hardware configuration registry
├── requirements.txt           # Verified package lockfile for Python 3.11 execution
├── README.md                  # Comprehensive architectural and theoretical documentation
├── summarizer/                # Domain AI Package (Abstracted Transformer Model Layer)
│   ├── __init__.py            # Clean API boundary exposing summarization engine & utility classes
│   ├── model.py               # Singleton PyTorch/Hugging Face BART sequence generator
│   └── utils.py               # Text preprocessing, execution timer, chunker, & analytics algorithms
│
├── templates/                 # Presentation Layer (Responsive Jinja2 / Bootstrap 5 templates)
│   ├── base.html              # Core Bootstrap 5 structural layout & Google font inclusions
│   ├── index.html             # Input web terminal with interactive client word counters
│   └── result.html            # Comparative presentation displaying metrics & summary analytics
│
├── static/                    # Frontend custom styling, micro-animations, and interactivity
│   ├── css/style.css          # Rich modern aesthetic styling (Glassmorphic dark UI accents)
│   ├── js/main.js             # Real-time interactive counters, asynchronous state loaders & clipboard copy
│   └── images/                # Asset storage
│
└── uploads/                   # Temporary filesystem workspace for incoming document processing
```

---

## 🧠 Model & Algorithmic Design: BART Architecture
**BART** unites the bidirectional encoding capabilities of **BERT** with the autoregressive left-to-right decoding mechanics of **GPT**. By masking and corrupting inputs during pre-training and tasking the autoregressive decoder with exact sequence reconstruction, BART exhibits exceptional capabilities in abstractive text summarization.

### Key Algorithmic Mitigations Implemented:
1. **Singleton Lazy Loading**: The neural model (~1.6 GB disk space / 2.5–4.0 GB RAM) is instantiated strictly once into server memory via a thread-safe Singleton design. This eliminates redundant RAM exhaustion across successive HTTP user requests.
2. **Context Window Chunking**: While standard BART attention limits context windows to **1024 positional tokens**, this architecture incorporates intelligent sentence-boundary segmentation to cleanly summarize documents extending well beyond typical token ceilings without context truncation.
3. **Hardware Acceleration Autodepreation**: Autonomously senses CUDA NVIDIA GPU computing power via PyTorch runtime heuristics; gracefully defaults to highly optimized multi-threaded CPU inference when dedicated graphic cores are unavailable.

---

## 📊 Reported Performance Metrics
On every summarization job, the application extracts and computes real-time analytical KPIs:
* **Original Word Count**: Exact count of tokens present in the source text submitted.
* **Summary Word Count**: Resulting sequence length produced by beam-search decoding.
* **Compression Percentage**: Mathematical data footprint reduction defined as:
  $$\text{Compression \%} = \left(1 - \frac{\text{Summary Words}}{\text{Original Words}}\right) \times 100$$
* **Processing Time**: Exact inference runtime precision calculated via hardware performance timers (`time.perf_counter()`), reported in decimal seconds.

---

## 🚀 Local Installation & Execution

### 1. Requirements & Prerequisites
* **Python 3.11+** installed on host OS.
* Sufficient memory (~4GB System RAM minimum for CPU inference; CUDA-compatible GPU recommended).

### 2. Virtual Environment Creation
```bash
# Clone or navigate to the repository directory
cd text-summarizer

# Create isolated Python virtual environment
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows Command Prompt
venv\Scripts\activate
```

### 3. Dependency Installation
```bash
# Upgrade pip to latest version and install project requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Running the Web Application Server
```bash
# Launch the Flask development backend server
python app.py
```
Upon startup, access the live dashboard by opening a modern web browser at:
👉 **http://127.0.0.1:5000/**

*(Note: During the very first summarization operation, Hugging Face Transformers will automatically download and cache the `facebook/bart-large-cnn` pre-trained checkpoint).*

---
## 👨‍💻 License & Academic Integrity
Developed to demonstrate academic software engineering paired with modern natural language processing transformations.

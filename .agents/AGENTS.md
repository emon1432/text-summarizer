# Project Guidelines: AI Text Summarizer

This document defines the core rules, constraints, and behavioral instructions for AI agents working on this workspace.

## 1. Project Context
- **Name**: AI Powered Text Summarization using Pre-trained Transformer Models.
- **Purpose**: Academic NLP Project. 
- **Tech Stack**: Python 3.11, Flask, PyTorch, Hugging Face Transformers (`facebook/bart-large-cnn`), Bootstrap 5.

## 2. Communication & Tone constraints
- **Language**: Use concise, professional, and academic language suitable for a university thesis defense.
- **Banned Terminology**: NEVER use exaggerated commercial/marketing buzzwords (e.g., "Enterprise", "Industrial Grade", "Mission Critical", "Production Ready", "Neural Engine", "High Performance Architecture"). Prefer simplicity over complexity.

## 3. Architecture Rules
- **Backend (Python)**:
  - Adhere strictly to PEP8 standards.
  - Follow the Clean Architecture pattern (separation of routes in `app.py`, logic in `summarizer/`).
  - Maintain the Singleton pattern in `TransformerSummarizer` (`summarizer/model.py`) to prevent multiple multi-GB model weights from exhausting system RAM.
  - Always validate and clamp hyperparameters (e.g., `max_length`, `min_length`, `num_beams`) before generation to prevent PyTorch inference crashes.
- **Frontend (Web)**:
  - Keep the UI simple and clean. Use standard Jinja2 syntax (e.g., `{{ variable }}` instead of `{{ variable:, }}`).
  - Maintain browser bfcache back-navigation recovery in `static/js/main.js` (using the `pageshow` event listener) so that loading spinners do not permanently lock the UI.

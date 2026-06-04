# NLP Evaluation Lab

An interactive Streamlit application for evaluating pre-trained transformer models on extractive question answering and abstractive summarization. The lab goes beyond generating outputs — it measures them using production-grade evaluation methodology and supports batch evaluation, multi-model comparison, and a structured production decision framework.

**Live demo:** [nlp-eval-lab-4undhkyatbnsxkrzbcktek.streamlit.app](https://nlp-eval-lab-4undhkyatbnsxkrzbcktek.streamlit.app/)
---

## Overview

Most NLP demos stop at showing model output. This lab asks the harder question: how do you know if the output is actually good?

The app implements evaluation metrics from scratch, makes every normalization step visible, flags what ROUGE misses, and turns the fine-tuning decision into a measurable, auditable process rather than an intuitive one.

---

## Features

### Extractive QA
- Span prediction over a context passage using encoder-only transformers
- **Exact Match (EM)** and **Token-F1** computed from scratch
- Full SQuAD-style normalization trace: lowercase → strip articles → strip punctuation → collapse whitespace
- Highlighted answer span in the original context
- **Batch evaluation** via CSV upload with aggregate metrics and per-example predictions download

### Abstractive Summarization
- Encoder–decoder generation with deterministic beam search
- **ROUGE-1**, **ROUGE-2**, **ROUGE-L** F1 scored against a reference summary
- **Faithfulness audit** — tracks key numeric values from source to summary, surfacing what ROUGE cannot detect
- **Corpus evaluation** via dual CSV upload (articles + references merged on `article_id`)

### Decision Matrix
- Five-factor interactive tool for the fine-tune vs. pre-trained inference decision
- Factors: labeled data availability, task specificity, compute budget, iteration speed, quality gap
- Weighted score (0–100) with plain-English reasoning grounded in each factor value
- Thresholds: < 35 → pre-trained · 35–60 → evaluate baseline first · > 60 → fine-tune

### Multi-Model Selection
- Three QA checkpoints switchable at runtime
- Three summarization checkpoints switchable at runtime
- Each model cached independently — switching does not re-download

### Three-Tier Inference
```
HuggingFace Inference API  →  Colab GPU (ngrok)  →  Local CPU
```
The app selects the fastest available backend automatically and falls back silently on failure.

---

## Models

### QA

| Model | Size | Notes |
|---|---|---|
| `distilbert-base-cased-distilled-squad` | 65MB | Default. Fast on CPU. SQuAD v1.1. |
| `deepset/roberta-base-squad2` | 125MB | Supports no-answer prediction. SQuAD v2.0. |
| `deepset/deberta-v3-base-squad2` | 180MB | Highest accuracy. Slower on CPU. |

### Summarization

| Model | Size | Notes |
|---|---|---|
| `sshleifer/distilbart-cnn-6-6` | 230MB | Default. Fast on CPU. CNN/DM distilled. |
| `facebook/bart-large-cnn` | 400MB | Best ROUGE quality. Full BART. |
| `google/pegasus-xsum` | 570MB | Concise abstractive summaries. |

---

## Evaluation Methodology

### QA
- Normalization: lowercase → strip standalone articles (a/an/the) → strip punctuation → collapse whitespace
- EM: 1 if normalized prediction equals normalized gold, else 0
- Token-F1: harmonic mean of token-level precision and recall after normalization
- Edge cases: both empty → 1.0 · one empty → 0.0 · no overlap → 0.0
- Follows SQuAD v1.1 evaluation script conventions

### Summarization
- ROUGE-1/2/L F1 via `rouge-score` library
- `use_stemmer=True`
- `scorer.score(reference, predicted)` — reference is always first
- Generation: `do_sample=False`, `num_beams=4`, `no_repeat_ngram_size=3`

---

## Project Structure

```
nlp-eval-lab/
├── app.py              ← Streamlit UI — imports only, no logic
├── core/
│   ├── models.py       ← Model loading + three-tier inference
│   ├── metrics.py      ← EM, Token-F1, ROUGE, Decision Matrix score
│   ├── evaluator.py    ← Batch QA + corpus summarization evaluation
│   └── utils.py        ← Normalization, context highlighting, number extraction
├── data/
│   └── examples.py     ← Quick-load examples for all tabs
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/USERNAME/nlp-eval-lab.git
cd nlp-eval-lab
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

---

## Requirements

```
streamlit>=1.35.0
transformers>=4.41,<5.0
torch>=2.0,<3.0
rouge-score>=0.0.4
datasets>=2.19.0
scikit-learn>=1.4.0
pandas>=2.0
numpy>=1.26
requests>=2.28.0
python-dotenv>=1.0
huggingface_hub>=0.20
```

---

## License

MIT
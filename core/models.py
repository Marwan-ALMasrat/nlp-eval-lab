"""
core/models.py
──────────────
Model loading with dynamic model selection.
Supports two inference modes:
  - Local:  loads HuggingFace pipeline directly (slow on CPU)
  - Remote: sends requests to a FastAPI server running on Google Colab GPU via ngrok

Set COLAB_API_URL in Streamlit Secrets or .env to enable remote mode.
If COLAB_API_URL is not set, falls back to local inference automatically.
"""

import os

import requests
import streamlit as st
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from transformers.pipelines import QuestionAnsweringPipeline, SummarizationPipeline


# ─── Available models ────────────────────────────────────────────────────────

QA_MODELS = {
    "distilbert-base-cased-distilled-squad": {
        "label":       "DistilBERT · SQuAD v1.1 (fast, 65MB)",
        "description": "Encoder-only. Always returns an answer. Best for speed.",
    },
    "deepset/roberta-base-squad2": {
        "label":       "RoBERTa · SQuAD v2.0 (stronger, 125MB)",
        "description": "Supports 'no answer' prediction. Better on harder questions.",
    },
    "deepset/deberta-v3-base-squad2": {
        "label":       "DeBERTa · SQuAD v2.0 (best quality, 180MB)",
        "description": "Highest accuracy. Slower on CPU.",
    },
}

SUMM_MODELS = {
    "sshleifer/distilbart-cnn-6-6": {
        "label":       "DistilBART CNN (fast, 230MB)",
        "description": "Distilled BART. Fast on CPU. May repeat n-grams.",
    },
    "facebook/bart-large-cnn": {
        "label":       "BART Large CNN (best quality, 400MB)",
        "description": "Full BART fine-tuned on CNN/DM. Highest ROUGE scores.",
    },
    "google/pegasus-xsum": {
        "label":       "PEGASUS XSum (concise summaries, 570MB)",
        "description": "Trained on XSum. Produces shorter, more abstractive summaries.",
    },
}

DEFAULT_QA_MODEL   = "distilbert-base-cased-distilled-squad"
DEFAULT_SUMM_MODEL = "sshleifer/distilbart-cnn-6-6"


# ─── Remote mode detection ───────────────────────────────────────────────────

def _colab_url() -> str:
    """Return the Colab API URL from Streamlit Secrets or environment variable."""
    try:
        return st.secrets.get("COLAB_API_URL", "")
    except Exception:
        return os.environ.get("COLAB_API_URL", "")


def _is_remote() -> bool:
    return bool(_colab_url())


# ─── Local pipeline loaders ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading QA model…")
def _get_qa_pipeline_local(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForQuestionAnswering.from_pretrained(model_id)
    return QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)


@st.cache_resource(show_spinner="Loading summarization model…")
def _get_summ_pipeline_local(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return SummarizationPipeline(model=model, tokenizer=tokenizer)


# ─── Run functions ───────────────────────────────────────────────────────────

def run_qa(question: str, context: str, model_id: str = DEFAULT_QA_MODEL) -> dict:
    """
    Run extractive QA — remote (Colab GPU) or local (CPU) automatically.

    Returns:
        {
            "answer": str,
            "score":  float,
            "start":  int,
            "end":    int,
        }
    """
    if _is_remote():
        try:
            response = requests.post(
                f"{_colab_url()}/qa",
                json={"question": question, "context": context, "model_id": model_id},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.warning(f"Remote inference failed ({e}) — falling back to local.")

    pipeline = _get_qa_pipeline_local(model_id)
    return pipeline(question=question, context=context)


def run_summarization(text: str, model_id: str = DEFAULT_SUMM_MODEL) -> str:
    """
    Run abstractive summarization — remote (Colab GPU) or local (CPU) automatically.

    Generation settings:
        max_length=80, min_length=20, do_sample=False,
        num_beams=4, no_repeat_ngram_size=3

    Returns the summary string.
    """
    if _is_remote():
        try:
            response = requests.post(
                f"{_colab_url()}/summarize",
                json={"text": text, "model_id": model_id},
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["summary"]
        except Exception as e:
            st.warning(f"Remote inference failed ({e}) — falling back to local.")

    pipeline = _get_summ_pipeline_local(model_id)
    out = pipeline(
        text,
        max_length=80,
        min_length=20,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
    )
    return out[0]["summary_text"]
"""
core/models.py
──────────────
Model loading with dynamic model selection.
Supports three inference modes:
  - HuggingFace API: fastest, no local download, requires HF_TOKEN
  - Local:           loads HuggingFace pipeline directly (slow on CPU)
  - Remote (Colab):  sends requests to FastAPI server via ngrok

Priority: HuggingFace API → Colab Remote → Local CPU
"""

import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from transformers.pipelines import QuestionAnsweringPipeline, SummarizationPipeline

# ─── Load .env for local development ─────────────────────────────────────────
load_dotenv()


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

_NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}
_HF_API_URL    = "https://api-inference.huggingface.co/models"


# ─── Mode detection ──────────────────────────────────────────────────────────

def _hf_token() -> str:
    try:
        return st.secrets.get("HF_TOKEN", "")
    except Exception:
        return os.environ.get("HF_TOKEN", "")


def _colab_url() -> str:
    try:
        return st.secrets.get("COLAB_API_URL", "")
    except Exception:
        return os.environ.get("COLAB_API_URL", "")


def _is_hf_api() -> bool:
    return bool(_hf_token())


def _is_remote() -> bool:
    return bool(_colab_url())


# ─── Token check (runs once at startup) ──────────────────────────────────────

def _check_token() -> None:
    """Warn in the UI if no HF_TOKEN is found."""
    if not _hf_token():
        st.warning("⚠️ HF_TOKEN غير موجود — سيعمل محلياً فقط (بطيء)")


# ─── HuggingFace API inference ───────────────────────────────────────────────

def _hf_headers() -> dict:
    return {"Authorization": f"Bearer {_hf_token()}"}


def _hf_qa(question: str, context: str, model_id: str) -> dict:
    """Call HuggingFace Inference API for QA."""
    for attempt in range(3):
        response = requests.post(
            f"{_HF_API_URL}/{model_id}",
            headers=_hf_headers(),
            json={"inputs": {"question": question, "context": context}},
            timeout=30,
        )
        data = response.json()

        # Model loading — wait and retry
        if isinstance(data, dict) and "error" in data and "loading" in data.get("error", "").lower():
            wait = data.get("estimated_time", 20)
            st.info(f"Model loading on HuggingFace servers — retrying in {int(wait)}s…")
            time.sleep(wait)
            continue

        response.raise_for_status()
        return {
            "answer": data.get("answer", ""),
            "score":  data.get("score", 0.0),
            "start":  data.get("start", 0),
            "end":    data.get("end", 0),
        }
    raise RuntimeError("HuggingFace API unavailable after 3 attempts.")


def _hf_summarize(text: str, model_id: str) -> str:
    """Call HuggingFace Inference API for summarization."""
    for attempt in range(3):
        response = requests.post(
            f"{_HF_API_URL}/{model_id}",
            headers=_hf_headers(),
            json={
                "inputs": text,
                "parameters": {
                    "max_length": 80,
                    "min_length": 20,
                    "num_beams":  4,
                    "no_repeat_ngram_size": 3,
                },
            },
            timeout=60,
        )
        data = response.json()

        if isinstance(data, dict) and "error" in data and "loading" in data.get("error", "").lower():
            wait = data.get("estimated_time", 20)
            st.info(f"Model loading on HuggingFace servers — retrying in {int(wait)}s…")
            time.sleep(wait)
            continue

        response.raise_for_status()
        return data[0]["summary_text"]
    raise RuntimeError("HuggingFace API unavailable after 3 attempts.")


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


# ─── Public run functions ────────────────────────────────────────────────────

def run_qa(question: str, context: str, model_id: str = DEFAULT_QA_MODEL) -> dict:
    """
    Run extractive QA.
    Priority: HuggingFace API → Colab Remote → Local CPU

    Returns:
        {
            "answer": str,
            "score":  float,
            "start":  int,
            "end":    int,
        }
    """
    # 1 — HuggingFace API
    if _is_hf_api():
        try:
            return _hf_qa(question, context, model_id)
        except Exception as e:
            st.warning(f"HuggingFace API failed ({e}) — trying next mode.")

    # 2 — Colab Remote
    if _is_remote():
        try:
            response = requests.post(
                f"{_colab_url()}/qa",
                json={"question": question, "context": context, "model_id": model_id},
                headers=_NGROK_HEADERS,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.warning(f"Remote inference failed ({e}) — falling back to local.")

    # 3 — Local CPU
    pipeline = _get_qa_pipeline_local(model_id)
    return pipeline(question=question, context=context)


def run_summarization(text: str, model_id: str = DEFAULT_SUMM_MODEL) -> str:
    """
    Run abstractive summarization.
    Priority: HuggingFace API → Colab Remote → Local CPU

    Generation settings:
        max_length=80, min_length=20, do_sample=False,
        num_beams=4, no_repeat_ngram_size=3

    Returns the summary string.
    """
    # 1 — HuggingFace API
    if _is_hf_api():
        try:
            return _hf_summarize(text, model_id)
        except Exception as e:
            st.warning(f"HuggingFace API failed ({e}) — trying next mode.")

    # 2 — Colab Remote
    if _is_remote():
        try:
            response = requests.post(
                f"{_colab_url()}/summarize",
                json={"text": text, "model_id": model_id},
                headers=_NGROK_HEADERS,
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["summary"]
        except Exception as e:
            st.warning(f"Remote inference failed ({e}) — falling back to local.")

    # 3 — Local CPU
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
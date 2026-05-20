"""
core/models.py
──────────────
Model loading with dynamic model selection.
Priority: Groq API → HuggingFace API → Local CPU
الخيار يُقرأ من st.session_state["use_groq"] الذي يضبطه المستخدم من الـ sidebar.
"""

import os
import re
import json
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

_HF_API_URL   = "https://router.huggingface.co/hf-inference/models"
_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL   = "llama-3.1-8b-instant"


# ─── Credentials ─────────────────────────────────────────────────────────────

def _groq_key() -> str:
    try:
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")


def _hf_token() -> str:
    try:
        return st.secrets.get("HF_TOKEN", "")
    except Exception:
        return os.environ.get("HF_TOKEN", "")


def _is_groq() -> bool:
    return bool(_groq_key())


def _is_hf_api() -> bool:
    return bool(_hf_token())


def _user_wants_groq() -> bool:
    """Returns True if user selected Groq in the sidebar AND key exists."""
    return st.session_state.get("use_groq", True) and _is_groq()


# ─── Groq API ────────────────────────────────────────────────────────────────

def _groq_headers() -> dict:
    return {
        "Authorization": f"Bearer {_groq_key()}",
        "Content-Type": "application/json",
    }


def _groq_qa(question: str, context: str) -> dict:
    prompt = f"""You are a precise question-answering assistant.
Extract the answer to the question from the context below.
Return ONLY a JSON object with these fields:
- "answer": the exact answer string extracted from the context
- "score": confidence float between 0 and 1

Context: {context}

Question: {question}

Respond with JSON only, no explanation."""

    response = requests.post(
        _GROQ_API_URL,
        headers=_groq_headers(),
        json={
            "model": _GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 200,
        },
        timeout=15,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"].strip()
    content = re.sub(r"```json|```", "", content).strip()
    data    = json.loads(content)

    answer = data.get("answer", "")
    score  = float(data.get("score", 0.9))
    start  = context.find(answer)
    if start == -1:
        start = 0
    end = start + len(answer)

    return {"answer": answer, "score": score, "start": start, "end": end}


def _groq_summarize(text: str) -> str:
    prompt = f"""Summarize the following article in 2-3 concise sentences.
Return only the summary, no preamble or explanation.

Article:
{text}

Summary:"""

    response = requests.post(
        _GROQ_API_URL,
        headers=_groq_headers(),
        json={
            "model": _GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 150,
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


# ─── HuggingFace API ─────────────────────────────────────────────────────────

def _hf_headers() -> dict:
    return {"Authorization": f"Bearer {_hf_token()}"}


def _hf_qa(question: str, context: str, model_id: str) -> dict:
    for attempt in range(3):
        response = requests.post(
            f"{_HF_API_URL}/{model_id}",
            headers=_hf_headers(),
            json={"inputs": {"question": question, "context": context}},
            timeout=30,
        )
        data = response.json()
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


# ─── Local pipelines ─────────────────────────────────────────────────────────

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
    Priority (based on sidebar selection):
      Groq selected  → Groq → fallback HF → fallback Local
      Local selected → Local directly
    """
    if _user_wants_groq():
        try:
            return _groq_qa(question, context)
        except Exception as e:
            st.warning(f"Groq API failed ({e}) — trying HuggingFace.")

    if _is_hf_api():
        try:
            return _hf_qa(question, context, model_id)
        except Exception as e:
            st.warning(f"HuggingFace API failed ({e}) — falling back to local.")

    pipeline = _get_qa_pipeline_local(model_id)
    return pipeline(question=question, context=context)


def run_summarization(text: str, model_id: str = DEFAULT_SUMM_MODEL) -> str:
    """
    Run abstractive summarization.
    Priority (based on sidebar selection):
      Groq selected  → Groq → fallback HF → fallback Local
      Local selected → Local directly
    """
    if _user_wants_groq():
        try:
            return _groq_summarize(text)
        except Exception as e:
            st.warning(f"Groq API failed ({e}) — trying HuggingFace.")

    if _is_hf_api():
        try:
            return _hf_summarize(text, model_id)
        except Exception as e:
            st.warning(f"HuggingFace API failed ({e}) — falling back to local.")

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
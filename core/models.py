"""
core/models.py
──────────────
Model loading with dynamic model selection.
Priority: Groq API (with key rotation) → Gemini 2.0 Flash → HuggingFace API → Local CPU
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

_HF_API_URL    = "https://router.huggingface.co/hf-inference/models"
_GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL    = "llama-3.1-8b-instant"
_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


# ─── Groq Key Rotation ───────────────────────────────────────────────────────

def _load_groq_keys() -> list:
    """
    Loads all available Groq keys from Secrets or .env.
    Supports:
      GROQ_API_KEY        (single key — backwards compatible)
      GROQ_API_KEY_1 ... GROQ_API_KEY_N  (multiple keys)
    """
    keys = []
    # Numbered keys: GROQ_API_KEY_1, _2, ...
    for i in range(1, 20):
        try:
            k = st.secrets.get(f"GROQ_API_KEY_{i}", "")
        except Exception:
            k = os.environ.get(f"GROQ_API_KEY_{i}", "")
        if k:
            keys.append(k)

    # Single unnumbered key as fallback
    if not keys:
        try:
            k = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            k = os.environ.get("GROQ_API_KEY", "")
        if k:
            keys.append(k)

    return keys


def _get_current_key_index() -> int:
    """Returns current key index from session_state."""
    return st.session_state.get("groq_key_index", 0)


def _rotate_key(keys: list) -> int:
    """Moves to the next key and returns the new index."""
    current = _get_current_key_index()
    next_index = (current + 1) % len(keys)
    st.session_state["groq_key_index"] = next_index
    return next_index


def _is_groq() -> bool:
    return len(_load_groq_keys()) > 0


def _is_hf_api() -> bool:
    try:
        return bool(st.secrets.get("HF_TOKEN", ""))
    except Exception:
        return bool(os.environ.get("HF_TOKEN", ""))


def _user_wants_groq() -> bool:
    return st.session_state.get("use_groq", True) and _is_groq()


def _user_wants_gemini() -> bool:
    return st.session_state.get("use_gemini", False) and _is_gemini()


# ─── Gemini helpers ───────────────────────────────────────────────────────────

def _gemini_key() -> str:
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return os.environ.get("GEMINI_API_KEY", "")


def _is_gemini() -> bool:
    return bool(_gemini_key())


def _gemini_post(prompt: str, timeout: int = 20) -> str:
    """POST to Gemini 2.0 Flash and return the text response."""
    key = _gemini_key()
    if not key:
        raise RuntimeError("No GEMINI_API_KEY found in Secrets.")

    response = requests.post(
        f"{_GEMINI_API_URL}?key={key}",
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 300},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _gemini_qa(question: str, context: str) -> dict:
    prompt = f"""You are a precise question-answering assistant.
Extract the answer to the question from the context below.
Return ONLY a JSON object with these fields:
- "answer": the exact answer string extracted from the context
- "score": confidence float between 0 and 1

Context: {context}

Question: {question}

Respond with JSON only, no explanation."""

    content = _gemini_post(prompt)
    content = re.sub(r"```json|```", "", content).strip()
    result  = json.loads(content)

    answer = result.get("answer", "")
    score  = float(result.get("score", 0.9))
    start  = context.find(answer)
    if start == -1:
        start = 0
    end = start + len(answer)
    return {"answer": answer, "score": score, "start": start, "end": end}


def _gemini_summarize(text: str) -> str:
    prompt = f"""Summarize the following article in 2-3 concise sentences.
Return only the summary, no preamble or explanation.

Article:
{text}

Summary:"""

    return _gemini_post(prompt, timeout=30)

def _groq_post(payload: dict, timeout: int = 15) -> dict:
    """
    POST to Groq with:
    - Automatic key rotation on 429
    - Cycles through all keys before waiting
    - Exponential backoff only after a full round
    """
    keys = _load_groq_keys()
    if not keys:
        raise RuntimeError("No GROQ_API_KEY found in Secrets.")

    attempt      = 0
    round_num    = 0
    wait         = 2.0
    MAX_ROUNDS   = 10

    while round_num < MAX_ROUNDS:
        idx      = _get_current_key_index() % len(keys)
        key      = keys[idx]
        attempt += 1

        response = requests.post(
            _GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )

        if response.status_code == 429:
            new_idx = _rotate_key(keys)

            # Completed a full round through all keys
            if new_idx == 0:
                round_num += 1
                retry_after = response.headers.get("Retry-After", "")
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        pass
                st.info(
                    f"⏳ All keys hit rate limit (round {round_num}/{MAX_ROUNDS}) — "
                    f"waiting {wait:.0f}s before retrying…"
                )
                time.sleep(wait)
                wait = min(wait * 2, 60)
            else:
                # New key available — switch immediately
                if len(keys) > 1:
                    st.info(f"⚡ Key {idx+1} hit rate limit — switching to key {new_idx+1}/{len(keys)}…")
                time.sleep(0.3)
            continue

        response.raise_for_status()
        return response.json()

    raise RuntimeError(
        f"Groq: exhausted {MAX_ROUNDS} full rounds across {len(keys)} keys. "
        "Wait a few minutes or add more API keys."
    )


# ─── Groq inference ──────────────────────────────────────────────────────────

def _groq_qa(question: str, context: str) -> dict:
    prompt = f"""You are a precise question-answering assistant.
Extract the answer to the question from the context below.
Return ONLY a JSON object with these fields:
- "answer": the exact answer string extracted from the context
- "score": confidence float between 0 and 1

Context: {context}

Question: {question}

Respond with JSON only, no explanation."""

    data    = _groq_post({
        "model": _GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 200,
    })
    content = data["choices"][0]["message"]["content"].strip()
    content = re.sub(r"```json|```", "", content).strip()
    result  = json.loads(content)

    answer = result.get("answer", "")
    score  = float(result.get("score", 0.9))
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

    data = _groq_post({
        "model": _GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 150,
    })
    return data["choices"][0]["message"]["content"].strip()


# ─── HuggingFace API ─────────────────────────────────────────────────────────

def _hf_token() -> str:
    try:
        return st.secrets.get("HF_TOKEN", "")
    except Exception:
        return os.environ.get("HF_TOKEN", "")


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
            st.info(f"Model loading on HuggingFace — retrying in {int(wait)}s…")
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
                    "max_length": 80, "min_length": 20,
                    "num_beams": 4, "no_repeat_ngram_size": 3,
                },
            },
            timeout=60,
        )
        data = response.json()
        if isinstance(data, dict) and "error" in data and "loading" in data.get("error", "").lower():
            wait = data.get("estimated_time", 20)
            st.info(f"Model loading on HuggingFace — retrying in {int(wait)}s…")
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
    if _user_wants_groq():
        try:
            return _groq_qa(question, context)
        except RuntimeError as e:
            st.warning(str(e))
        except Exception as e:
            st.warning(f"Groq failed ({e}) — trying Gemini…")

    if _user_wants_gemini() or _is_gemini():
        try:
            return _gemini_qa(question, context)
        except Exception as e:
            st.warning(f"Gemini failed ({e}) — falling back to HuggingFace.")

    if _is_hf_api():
        try:
            return _hf_qa(question, context, model_id)
        except Exception as e:
            st.warning(f"HuggingFace failed ({e}) — falling back to local.")

    pipeline = _get_qa_pipeline_local(model_id)
    return pipeline(question=question, context=context)


def run_summarization(text: str, model_id: str = DEFAULT_SUMM_MODEL) -> str:
    if _user_wants_groq():
        try:
            return _groq_summarize(text)
        except RuntimeError as e:
            st.warning(str(e))
        except Exception as e:
            st.warning(f"Groq failed ({e}) — trying Gemini…")

    if _user_wants_gemini() or _is_gemini():
        try:
            return _gemini_summarize(text)
        except Exception as e:
            st.warning(f"Gemini failed ({e}) — falling back to HuggingFace.")

    if _is_hf_api():
        try:
            return _hf_summarize(text, model_id)
        except Exception as e:
            st.warning(f"HuggingFace failed ({e}) — falling back to local.")

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
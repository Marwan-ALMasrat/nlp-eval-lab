"""
core/models.py
──────────────
Model loading — the only file that imports from `transformers`.
All pipelines are cached at the module level via Streamlit's cache_resource
so weights are downloaded once and reused across reruns.

Models used (per Module 7 Week B reading):
  QA:   distilbert-base-cased-distilled-squad   (~65M params, SQuAD v1.1)
  Summ: sshleifer/distilbart-cnn-6-6             (~230MB, CNN/DM distilled BART)
"""

import streamlit as st
from transformers import pipeline


QA_MODEL_ID   = "distilbert-base-cased-distilled-squad"
SUMM_MODEL_ID = "sshleifer/distilbart-cnn-6-6"


@st.cache_resource(show_spinner="Loading QA model — distilbert-base-cased-distilled-squad …")
def get_qa_pipeline():
    """
    Return a Hugging Face question-answering pipeline.

    Model: distilbert-base-cased-distilled-squad
    Architecture: encoder-only (DistilBERT) + QA head (start/end logits).
    Training data: SQuAD v1.1 — every question has an answer in the context.
    """
    return pipeline("question-answering", model=QA_MODEL_ID)


@st.cache_resource(show_spinner="Loading summarization model — sshleifer/distilbart-cnn-6-6 …")
def get_summ_pipeline():
    """
    Return a Hugging Face summarization pipeline.

    Model: sshleifer/distilbart-cnn-6-6
    Architecture: encoder–decoder (distilled BART fine-tuned on CNN/Daily Mail).
    Known quirk: prone to n-gram repetition without no_repeat_ngram_size guard.
    """
    return pipeline("summarization", model=SUMM_MODEL_ID)


def run_qa(question: str, context: str) -> dict:
    """
    Run the QA pipeline and return the raw result dict.

    Returns:
        {
            "answer": str,    # verbatim substring of context
            "score":  float,  # joint start+end probability (confidence signal)
            "start":  int,    # character offset in context
            "end":    int,    # character offset in context
        }
    """
    qa = get_qa_pipeline()
    return qa(question=question, context=context)


def run_summarization(article: str) -> str:
    """
    Run the summarization pipeline with the generation parameters from the reading.

    Generation settings (Section 6):
        max_length=80, min_length=20, do_sample=False,
        num_beams=4, no_repeat_ngram_size=3

    Returns the summary string.
    """
    summ = get_summ_pipeline()
    out  = summ(
        article,
        max_length=80,
        min_length=20,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
    )
    return out[0]["summary_text"]
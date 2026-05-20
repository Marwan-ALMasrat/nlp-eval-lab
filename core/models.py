"""
core/models.py
──────────────
Model loading with dynamic model selection.
All pipelines are cached per model_id via Streamlit's cache_resource.
"""

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
        "label": "DistilBERT · SQuAD v1.1 (fast, 65MB)",
        "description": "Encoder-only. Always returns an answer. Best for speed.",
    },
    "deepset/roberta-base-squad2": {
        "label": "RoBERTa · SQuAD v2.0 (stronger, 125MB)",
        "description": "Supports 'no answer' prediction. Better on harder questions.",
    },
    "deepset/deberta-v3-base-squad2": {
        "label": "DeBERTa · SQuAD v2.0 (best quality, 180MB)",
        "description": "Highest accuracy. Slower on CPU.",
    },
}

SUMM_MODELS = {
    "sshleifer/distilbart-cnn-6-6": {
        "label": "DistilBART CNN (fast, 230MB)",
        "description": "Distilled BART. Fast on CPU. May repeat n-grams.",
    },
    "facebook/bart-large-cnn": {
        "label": "BART Large CNN (best quality, 400MB)",
        "description": "Full BART fine-tuned on CNN/DM. Highest ROUGE scores.",
    },
    "google/pegasus-xsum": {
        "label": "PEGASUS XSum (concise summaries, 570MB)",
        "description": "Trained on XSum. Produces shorter, more abstractive summaries.",
    },
}

# Default model IDs
DEFAULT_QA_MODEL   = "distilbert-base-cased-distilled-squad"
DEFAULT_SUMM_MODEL = "sshleifer/distilbart-cnn-6-6"


# ─── Pipeline loaders ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading QA model…")
def get_qa_pipeline(model_id: str = DEFAULT_QA_MODEL):
    """
    Return a QA pipeline for the given model_id.
    Cached per model_id — switching models downloads once then reuses.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForQuestionAnswering.from_pretrained(model_id)
    return QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)


@st.cache_resource(show_spinner="Loading summarization model…")
def get_summ_pipeline(model_id: str = DEFAULT_SUMM_MODEL):
    """
    Return a summarization pipeline for the given model_id.
    Cached per model_id — switching models downloads once then reuses.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return SummarizationPipeline(model=model, tokenizer=tokenizer)


# ─── Run functions ───────────────────────────────────────────────────────────

def run_qa(question: str, context: str, model_id: str = DEFAULT_QA_MODEL) -> dict:
    """
    Run the QA pipeline and return the raw result dict.

    Returns:
        {
            "answer": str,
            "score":  float,
            "start":  int,
            "end":    int,
        }
    """
    qa = get_qa_pipeline(model_id)
    return qa(question=question, context=context)


def run_summarization(article: str, model_id: str = DEFAULT_SUMM_MODEL) -> str:
    """
    Run the summarization pipeline with deterministic beam search.

    Generation settings:
        max_length=80, min_length=20, do_sample=False,
        num_beams=4, no_repeat_ngram_size=3

    Returns the summary string.
    """
    summ = get_summ_pipeline(model_id)
    out  = summ(
        article,
        max_length=80,
        min_length=20,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
    )
    return out[0]["summary_text"]
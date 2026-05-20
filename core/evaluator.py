"""
core/evaluator.py
─────────────────
Batch evaluation for both QA and Summarization.
No Streamlit imports. Pure logic only.
"""

import pandas as pd

from core.metrics import qa_metrics, rouge_scores
from core.models  import run_qa, run_summarization


def evaluate_batch(df: pd.DataFrame, progress_callback=None, model_id: str = "distilbert-base-cased-distilled-squad") -> dict:
    """
    Run the QA pipeline over every row in df and compute aggregate metrics.

    Args:
        df:                DataFrame with columns: qid, question, context, gold_answer
        progress_callback: optional callable(current, total) for progress updates
        model_id:          HuggingFace model ID for QA

    Returns:
        {
            "em":          float,
            "f1":          float,
            "n":           int,
            "predictions": list[dict],
        }
    """
    required = {"qid", "question", "context", "gold_answer"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    predictions = []
    em_scores   = []
    f1_scores   = []
    total       = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        question    = str(row["question"])
        context     = str(row["context"])
        gold_answer = str(row["gold_answer"])

        result           = run_qa(question=question, context=context, model_id=model_id)
        predicted_answer = result["answer"]

        m = qa_metrics(predicted_answer, gold_answer)

        em_scores.append(m["exact_match"])
        f1_scores.append(m["token_f1"])

        predictions.append({
            "qid":              str(row["qid"]),
            "question":         question,
            "context_excerpt":  context[:80],
            "gold_answer":      gold_answer,
            "predicted_answer": predicted_answer,
            "em":               m["exact_match"],
            "f1":               m["token_f1"],
        })

        if progress_callback:
            progress_callback(i + 1, total)

    n = len(predictions)
    return {
        "em":          sum(em_scores) / n if n else 0.0,
        "f1":          sum(f1_scores) / n if n else 0.0,
        "n":           n,
        "predictions": predictions,
    }


def evaluate_summaries(df: pd.DataFrame, progress_callback=None, model_id: str = "sshleifer/distilbart-cnn-6-6") -> dict:
    """
    Run the summarization pipeline over every row in df and compute ROUGE.

    Args:
        df:                DataFrame with columns: article_id, text, reference_summary
        progress_callback: optional callable(current, total) for progress updates
        model_id:          HuggingFace model ID for summarization

    Returns:
        {
            "rouge1":      float,
            "rouge2":      float,
            "rougeL":      float,
            "n":           int,
            "predictions": list[dict],
        }

    Each prediction dict:
        {
            "article_id":        str,
            "text_excerpt":      str,
            "reference_summary": str,
            "predicted_summary": str,
            "rouge1":            float,
            "rouge2":            float,
            "rougeL":            float,
        }
    """
    required = {"article_id", "text", "reference_summary"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    predictions = []
    r1_scores   = []
    r2_scores   = []
    rl_scores   = []
    total       = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        text      = str(row["text"])
        reference = str(row["reference_summary"])

        predicted = run_summarization(text, model_id=model_id)
        rouge     = rouge_scores(predicted=predicted, reference=reference)

        r1_scores.append(rouge["rouge1"]["fmeasure"])
        r2_scores.append(rouge["rouge2"]["fmeasure"])
        rl_scores.append(rouge["rougeL"]["fmeasure"])

        predictions.append({
            "article_id":        str(row["article_id"]),
            "text_excerpt":      text[:80],
            "reference_summary": reference,
            "predicted_summary": predicted,
            "rouge1":            rouge["rouge1"]["fmeasure"],
            "rouge2":            rouge["rouge2"]["fmeasure"],
            "rougeL":            rouge["rougeL"]["fmeasure"],
        })

        if progress_callback:
            progress_callback(i + 1, total)

    n = len(predictions)
    return {
        "rouge1":      sum(r1_scores) / n if n else 0.0,
        "rouge2":      sum(r2_scores) / n if n else 0.0,
        "rougeL":      sum(rl_scores) / n if n else 0.0,
        "n":           n,
        "predictions": predictions,
    }
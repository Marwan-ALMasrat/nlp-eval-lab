"""
core/evaluator.py
─────────────────
Batch QA evaluation over a DataFrame of examples.
No Streamlit imports. Pure logic only.

Expected DataFrame columns: qid, question, context, gold_answer
"""

import pandas as pd

from core.metrics import qa_metrics
from core.models  import run_qa


def evaluate_batch(df: pd.DataFrame, progress_callback=None) -> dict:
    """
    Run the QA pipeline over every row in df and compute aggregate metrics.

    Args:
        df:                DataFrame with columns: qid, question, context, gold_answer
        progress_callback: optional callable(current, total) for progress updates

    Returns:
        {
            "em":          float,   # mean exact match across all examples
            "f1":          float,   # mean token-F1 across all examples
            "n":           int,     # number of examples evaluated
            "predictions": list[dict],  # one dict per example
        }

    Each prediction dict:
        {
            "qid":               str,
            "question":          str,
            "context_excerpt":   str,   # first 80 chars of context
            "gold_answer":       str,
            "predicted_answer":  str,
            "em":                int,   # 0 or 1
            "f1":                float,
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

        result           = run_qa(question=question, context=context)
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
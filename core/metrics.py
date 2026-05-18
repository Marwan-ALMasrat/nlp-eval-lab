"""
core/metrics.py
───────────────
All NLP evaluation metrics used in Module 7.
Pure Python + rouge-score only. No Streamlit, no torch.

Methodology (document any deviation from these defaults):
  - QA:   SQuAD-style normalization before EM / Token-F1
  - ROUGE: rouge-score library, use_stemmer=True, F1 reported
           scorer.score(reference, predicted)  ← reference FIRST
"""

from collections import Counter

from rouge_score.rouge_scorer import RougeScorer

from core.utils import normalize_answer


# ── Extractive QA metrics ────────────────────────────────────────────────────

def exact_match(predicted: str, gold: str) -> int:
    """
    Return 1 if normalized predicted == normalized gold, else 0.
    Both strings are normalized with SQuAD-style normalization before comparison.
    """
    pred_norm, _ = normalize_answer(predicted)
    gold_norm, _ = normalize_answer(gold)
    return int(pred_norm == gold_norm)


def token_f1(predicted: str, gold: str) -> float:
    """
    Token-level F1 (partial credit) after SQuAD-style normalization.

    Treats each string as a bag of whitespace-split tokens.
    Overlap = multiset intersection of predicted and gold token bags.

    Edge cases (per SQuAD evaluation script convention):
      - Both empty → 1.0
      - One empty  → 0.0
      - No overlap → 0.0
    """
    pred_norm, _ = normalize_answer(predicted)
    gold_norm, _ = normalize_answer(gold)

    pt = pred_norm.split()
    gt = gold_norm.split()

    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0

    common  = Counter(pt) & Counter(gt)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pt)
    recall    = overlap / len(gt)
    return (2 * precision * recall) / (precision + recall)


def qa_metrics(predicted: str, gold: str) -> dict:
    """
    Convenience wrapper returning both EM and Token-F1 in one call.

    Returns:
        {
            "exact_match": int,       # 0 or 1
            "token_f1":    float,     # 0.0 – 1.0
            "pred_norm":   str,
            "gold_norm":   str,
            "pred_steps":  list,      # normalization trace
            "gold_steps":  list,
        }
    """
    pred_norm, pred_steps = normalize_answer(predicted)
    gold_norm, gold_steps = normalize_answer(gold)

    pt = pred_norm.split()
    gt = gold_norm.split()

    common  = Counter(pt) & Counter(gt)
    overlap = sum(common.values())

    if not pt and not gt:
        f1 = 1.0
    elif not pt or not gt:
        f1 = 0.0
    elif overlap == 0:
        f1 = 0.0
    else:
        prec = overlap / len(pt)
        rec  = overlap / len(gt)
        f1   = (2 * prec * rec) / (prec + rec)

    return {
        "exact_match": int(pred_norm == gold_norm),
        "token_f1":    round(f1, 4),
        "pred_norm":   pred_norm,
        "gold_norm":   gold_norm,
        "pred_steps":  pred_steps,
        "gold_steps":  gold_steps,
        "pred_tokens": len(pt),
        "gold_tokens": len(gt),
    }


# ── Summarization metrics ────────────────────────────────────────────────────

_SCORER = None  # module-level singleton to avoid re-init overhead


def _get_scorer() -> RougeScorer:
    global _SCORER
    if _SCORER is None:
        _SCORER = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return _SCORER


def rouge_scores(predicted: str, reference: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        predicted:  the generated summary
        reference:  the human reference summary

    Note: scorer.score(reference, predicted) — reference is the FIRST argument.
    use_stemmer=True matches the standard summarization-evaluation setting.

    Returns:
        {
            "rouge1": {"precision": float, "recall": float, "fmeasure": float},
            "rouge2": {...},
            "rougeL": {...},
        }
    """
    scorer = _get_scorer()
    raw    = scorer.score(reference, predicted)   # reference FIRST

    return {
        key: {
            "precision": round(raw[key].precision, 4),
            "recall":    round(raw[key].recall,    4),
            "fmeasure":  round(raw[key].fmeasure,  4),
        }
        for key in ("rouge1", "rouge2", "rougeL")
    }


# ── Decision matrix score ────────────────────────────────────────────────────

def decision_score(factor_values: list[int], weights: list[float]) -> int:
    """
    Compute a 0–100 fine-tuning signal score from five factor values (1–5)
    and their corresponding weights (must sum to 1.0).

    Higher score = stronger case for fine-tuning.
    Thresholds: < 35 → pre-trained; 35–60 → evaluate first; > 60 → fine-tune.
    """
    raw = sum(v * w for v, w in zip(factor_values, weights))
    return int(((raw - 1) / 4) * 100)


def decision_rationale(factor_values: list[int]) -> str:
    """
    Build a plain-English rationale string from the five slider values.
    Returns a single paragraph combining all triggered conditions.
    """
    data, spec, compute, speed, gap = factor_values
    parts = []

    if data <= 2:
        parts.append("Labeled data is scarce  fine-tuning requires thousands of examples.")
    elif data >= 4:
        parts.append("Sufficient labeled data is available to support fine-tuning.")

    if spec <= 2:
        parts.append("The task is generic  pre-trained models cover this distribution well.")
    elif spec >= 4:
        parts.append("The task is domain-specific  pre-trained baselines may underperform significantly.")

    if compute <= 2:
        parts.append("Compute is limited  fine-tuning (10–40 min per run, multiple HPO runs) may not be feasible.")
    elif compute >= 4:
        parts.append("Ample compute is available, making training runs feasible.")

    if speed <= 2:
        parts.append("Fast iteration is needed pre-trained inference can be swapped overnight vs. days per training run.")
    elif speed >= 4:
        parts.append("The project can accommodate multi-day training iterations.")

    if gap <= 2:
        parts.append("The pre-trained baseline already meets or nearly meets the quality target.")
    elif gap >= 4:
        parts.append("There is a significant quality gap  fine-tuning could close it by 10+ percentage points.")

    if not parts:
        parts.append(
            "All factors are at mid-range. Run the pre-trained baseline first "
            "and measure the quality gap before committing to fine-tuning."
        )

    return " ".join(parts)
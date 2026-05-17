"""
core/utils.py
─────────────
Pure-Python helpers with no dependency on Streamlit or any ML library.
These can be imported and unit-tested independently.
"""

import re
import string


# ── Answer normalization (SQuAD-style) ──────────────────────────────────────

def normalize_answer(s: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Apply SQuAD-style normalization and return (normalized_string, trace).

    Trace is a list of (step_label, value_after_step) tuples, useful for
    displaying the normalization pipeline step-by-step in the UI.

    Steps:
        1. lowercase
        2. strip articles  — remove standalone a / an / the
        3. strip punctuation — remove all string.punctuation characters
        4. collapse whitespace — split + rejoin
    """
    steps: list[tuple[str, str]] = [("original", s)]

    s = s.lower()
    steps.append(("lowercase", s))

    s = re.sub(r"\b(a|an|the)\b", " ", s)
    steps.append(("strip articles", s))

    s = "".join(ch for ch in s if ch not in string.punctuation)
    steps.append(("strip punctuation", s))

    s = " ".join(s.split())
    steps.append(("collapse whitespace", s))

    return s, steps


# ── Context highlighting ─────────────────────────────────────────────────────

def highlight_context(context: str, answer: str) -> str:
    """
    Return an HTML string where the answer span inside context is wrapped
    in a <mark> tag.  Falls back to plain context if the span is not found.
    """
    if not answer:
        return context

    idx = context.lower().find(answer.lower())
    if idx < 0:
        return context

    before = context[:idx]
    match  = context[idx : idx + len(answer)]
    after  = context[idx + len(answer):]
    return f"{before}<mark>{match}</mark>{after}"


# ── Number extraction (for faithfulness check) ───────────────────────────────

def extract_numbers(text: str) -> list[str]:
    """
    Return a deduplicated list of numeric tokens found in `text`.
    Captures integers, decimals, comma-formatted numbers, and trailing
    unit suffixes (%, x, X).

    Examples: "1,000", "5%", "12x", "0.95"
    """
    return list(set(re.findall(r"\b\d[\d,]*\.?\d*\b(?:%|x|X)?", text)))
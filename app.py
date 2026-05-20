"""
app.py
──────
Streamlit UI layer — the only file that imports streamlit.
All AI inference lives in core/models.py.
All metric computation lives in core/metrics.py.
All text helpers live in core/utils.py.
All examples live in data/examples.py.
Batch evaluation logic lives in core/evaluator.py.

Run:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from core.evaluator import evaluate_batch, evaluate_summaries
from core.metrics   import decision_rationale, decision_score, qa_metrics, rouge_scores
from core.models    import run_qa, run_summarization
from core.utils     import extract_numbers, highlight_context, normalize_answer
from data.examples  import DECISION_FACTORS, QA_EXAMPLES, SUMM_EXAMPLES


# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Evaluation Lab — Module 7 Week B",
    page_icon="🧪",
    layout="wide",
)

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.lab-header {
    display: flex; align-items: center; gap: 10px;
    padding: 0.5rem 0 1.2rem 0;
    border-bottom: 1px solid rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
}
.lab-title  { font-size: 20px; font-weight: 500; margin: 0; }
.lab-badge  { background:#eeedfe; color:#534ab7; font-size:11px; font-weight:500; padding:2px 10px; border-radius:20px; }
.lab-sub    { color:#888780; font-size:13px; margin-left:auto; }

.metric-card {
    background:#f1efe8; border-radius:8px;
    padding:10px 16px; text-align:center; flex:1;
}
.metric-label { font-size:11px; color:#888780; margin-bottom:4px; }
.metric-value { font-size:24px; font-weight:500; }
.metric-sub   { font-size:11px; color:#888780; margin-top:2px; }

.batch-metric-card {
    background:#eeedfe; border-radius:8px;
    padding:14px 16px; text-align:center; flex:1;
}

.summ-metric-card {
    background:#e1f5ee; border-radius:8px;
    padding:14px 16px; text-align:center; flex:1;
}

.answer-box {
    background:#e1f5ee; border:1px solid #0f6e56;
    border-radius:8px; padding:10px 14px;
    font-size:15px; font-weight:500; color:#0f6e56; margin:8px 0;
}
.norm-trace {
    background:#f1efe8; border-radius:8px;
    padding:10px 12px; font-family:monospace; font-size:12px; margin:6px 0;
}
.rouge-row   { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
.rouge-label { font-size:12px; width:68px; color:#5f5e5a; flex-shrink:0; }
.rouge-track { flex:1; height:6px; background:#f1efe8; border-radius:3px; overflow:hidden; }
.rouge-fill  { height:100%; background:#534ab7; border-radius:3px; }
.rouge-val   { font-size:12px; font-weight:500; width:40px; text-align:right; }

.faith-found   { background:#e1f5ee; color:#0f6e56; border-radius:4px; padding:2px 8px; font-size:12px; font-weight:500; display:inline-block; margin:2px; }
.faith-missing { background:#faece7; color:#993c1d; border-radius:4px; padding:2px 8px; font-size:12px; font-weight:500; display:inline-block; margin:2px; }

.ctx-text { font-size:13px; line-height:1.8; background:#f1efe8; border-radius:8px; padding:10px 12px; }
mark { background:#faeeda; color:#854f0b; border-radius:3px; padding:1px 3px; font-weight:500; }

.section-label { font-size:11px; font-weight:500; text-transform:uppercase; letter-spacing:0.06em; color:#888780; margin-bottom:4px; }

.rec-pretrained { background:#e1f5ee; border-radius:12px; padding:1rem 1.25rem; }
.rec-borderline { background:#faeeda; border-radius:12px; padding:1rem 1.25rem; }
.rec-finetune   { background:#eeedfe; border-radius:12px; padding:1rem 1.25rem; }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lab-header">
  <span style="font-size:24px">🧪</span>
  <p class="lab-title">NLP Evaluation</p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab_qa, tab_summ, tab_decision = st.tabs([
    "🔍  Extractive QA",
    "📄  Summarization",
    "⚖️  Decision Matrix",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXTRACTIVE QA
# ════════════════════════════════════════════════════════════════════════════
with tab_qa:
    st.markdown("#### Model: `distilbert-base-cased-distilled-squad` · Hugging Face pipeline")
    st.caption(
        "Encoder-only (DistilBERT) + QA head. "
        "Span prediction: per-token start/end logits over the context. "
        "Fine-tuned on SQuAD v1.1 — every question has an answer in the context."
    )

    mode = st.radio(
        "Mode",
        ["Single Question", "Batch Evaluation"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SINGLE QUESTION MODE
    # ════════════════════════════════════════════════════════════════════════
    if mode == "Single Question":

        col_inp, col_ex = st.columns([3, 1])

        with col_ex:
            st.markdown('<div class="section-label">Quick examples</div>', unsafe_allow_html=True)
            ex_choice = st.radio("", list(QA_EXAMPLES.keys()), label_visibility="collapsed")
            if st.button("Load example", key="qa_load"):
                st.session_state["qa_context"]  = QA_EXAMPLES[ex_choice]["context"]
                st.session_state["qa_question"] = QA_EXAMPLES[ex_choice]["question"]
                st.session_state["qa_gold"]     = QA_EXAMPLES[ex_choice]["gold"]

        with col_inp:
            context  = st.text_area("Context passage",               key="qa_context",  height=130,
                                     placeholder="Paste the context passage here…")
            question = st.text_input("Question",                     key="qa_question",
                                      placeholder="Ask a question answerable from the context…")
            gold     = st.text_input("Gold answer (for evaluation)", key="qa_gold",
                                      placeholder="The reference answer span…")

        if st.button("▶  Run extractive QA", type="primary", key="qa_run"):
            if not context or not question:
                st.warning("Please fill in the context and question.")
            else:
                with st.spinner("Running QA pipeline…"):
                    result = run_qa(question=question, context=context)

                answer = result["answer"]
                score  = result["score"]
                start  = result["start"]
                end    = result["end"]

                st.markdown("##### Model answer")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                st.caption(
                    f"Confidence score: **{score:.4f}** · character span [{start}, {end}] · "
                    "verbatim substring of context via start/end logit prediction."
                )

                st.markdown("##### Context with highlighted span")
                st.markdown(
                    f'<div class="ctx-text">{highlight_context(context, answer)}</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("##### SQuAD-style normalization trace")
                st.caption("lowercase → strip articles (a/an/the) → strip punctuation → collapse whitespace")

                _, pred_steps = normalize_answer(answer)
                c1, c2 = st.columns(2)

                with c1:
                    html = "<div class='norm-trace'><b style='font-size:11px;color:#888780'>PREDICTION</b>"
                    for label, val in pred_steps:
                        html += f"<br><span style='color:#888780'>{label}:</span> &quot;{val}&quot;"
                    html += "</div>"
                    st.markdown(html, unsafe_allow_html=True)

                with c2:
                    if gold:
                        _, gold_steps = normalize_answer(gold)
                        html = "<div class='norm-trace'><b style='font-size:11px;color:#888780'>GOLD</b>"
                        for label, val in gold_steps:
                            html += f"<br><span style='color:#888780'>{label}:</span> &quot;{val}&quot;"
                        html += "</div>"
                        st.markdown(html, unsafe_allow_html=True)
                    else:
                        st.info("Provide a gold answer to see its normalization trace.")

                if gold:
                    m = qa_metrics(answer, gold)

                    st.markdown("##### Evaluation metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    em_color = "green" if m["exact_match"] else "red"
                    for col, lbl, val, sub in [
                        (col1, "Exact Match", f'<span style="color:{em_color}">{m["exact_match"]}</span>',
                                              "✓ exact" if m["exact_match"] else "✗ no match"),
                        (col2, "Token-F1",    f'{m["token_f1"]:.3f}', "partial credit"),
                        (col3, "Pred tokens", str(m["pred_tokens"]),  "after norm"),
                        (col4, "Gold tokens", str(m["gold_tokens"]),  "after norm"),
                    ]:
                        with col:
                            col.markdown(
                                f'<div class="metric-card">'
                                f'<div class="metric-label">{lbl}</div>'
                                f'<div class="metric-value">{val}</div>'
                                f'<div class="metric-sub">{sub}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                    gap = m["token_f1"] - m["exact_match"]
                    if gap > 0.15:
                        st.info(
                            f"Token-F1 ({m['token_f1']:.2f}) is significantly higher than "
                            f"EM ({m['exact_match']}) — the model found the right region "
                            "but span boundaries differ from the gold answer."
                        )
                    elif m["exact_match"] == 1:
                        st.success(f"EM = 1 and F1 = {m['token_f1']:.2f} — perfect span match.")
                    else:
                        st.error(
                            f"EM = 0 and F1 = {m['token_f1']:.2f} — the predicted span "
                            "has little or no overlap with the gold answer."
                        )
                else:
                    st.info("Provide a gold answer to compute EM and Token-F1.")

    # ════════════════════════════════════════════════════════════════════════
    # BATCH EVALUATION MODE
    # ════════════════════════════════════════════════════════════════════════
    else:
        st.markdown("##### Upload a CSV to evaluate")
        st.caption(
            "Required columns: `qid`, `question`, `context`, `gold_answer`. "
            "Every gold answer must be a verbatim substring of its context (SQuAD v1.1 convention)."
        )

        uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_upload")

        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                df = None

            if df is not None:
                st.markdown(f"**{len(df)} examples loaded** — preview:")
                st.dataframe(df.head(5), use_container_width=True)

                max_rows = st.slider(
                    "Rows to evaluate (reduce for a quick test)",
                    min_value=1,
                    max_value=len(df),
                    value=min(50, len(df)),
                    key="batch_limit",
                )

                if st.button("▶  Run batch evaluation", type="primary", key="batch_run"):
                    df_eval      = df.head(max_rows).reset_index(drop=True)
                    progress_bar = st.progress(0)
                    status_text  = st.empty()

                    def update_progress(current, total):
                        progress_bar.progress(current / total)
                        status_text.caption(f"Evaluating example {current} / {total}…")

                    with st.spinner("Running batch evaluation…"):
                        try:
                            results = evaluate_batch(df_eval, progress_callback=update_progress)
                        except ValueError as e:
                            st.error(str(e))
                            results = None

                    progress_bar.empty()
                    status_text.empty()

                    if results:
                        st.markdown("##### Aggregate results")
                        c1, c2, c3 = st.columns(3)
                        for col, lbl, val, sub in [
                            (c1, "Aggregate EM", f'{results["em"]:.3f}', f'{int(results["em"]*100)}% exact matches'),
                            (c2, "Aggregate F1", f'{results["f1"]:.3f}', "mean token-F1"),
                            (c3, "Examples",     str(results["n"]),      "evaluated"),
                        ]:
                            with col:
                                col.markdown(
                                    f'<div class="batch-metric-card">'
                                    f'<div class="metric-label">{lbl}</div>'
                                    f'<div class="metric-value">{val}</div>'
                                    f'<div class="metric-sub">{sub}</div>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                        gap = results["f1"] - results["em"]
                        if gap > 0.15:
                            st.info(
                                f"F1 ({results['f1']:.2f}) is significantly higher than "
                                f"EM ({results['em']:.2f}) — the model finds the right region "
                                "but often misses exact span boundaries."
                            )
                        elif results["em"] > 0.7:
                            st.success("Strong performance — EM above 0.70 on this dataset.")
                        else:
                            st.warning(
                                "EM is below 0.70. Consider whether the task distribution "
                                "matches SQuAD v1.1 training data, or whether fine-tuning would help."
                            )

                        st.markdown("##### Per-example predictions")
                        pred_df = pd.DataFrame(results["predictions"])
                        st.dataframe(pred_df, use_container_width=True)

                        st.download_button(
                            label="⬇  Download predictions CSV",
                            data=pred_df.to_csv(index=False).encode("utf-8"),
                            file_name="qa_predictions.csv",
                            mime="text/csv",
                            key="batch_download",
                        )

                        st.caption(
                            "Methodology: SQuAD-style normalization applied before EM and Token-F1. "
                            "Model: `distilbert-base-cased-distilled-squad`."
                        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — SUMMARIZATION
# ════════════════════════════════════════════════════════════════════════════
with tab_summ:
    st.markdown("#### Model: `sshleifer/distilbart-cnn-6-6` · Hugging Face pipeline")
    st.caption(
        "Encoder–decoder (distilled BART fine-tuned on CNN/Daily Mail). "
        "Generation: `num_beams=4`, `do_sample=False`, `no_repeat_ngram_size=3`."
    )

    summ_mode = st.radio(
        "Summarization Mode",
        ["Single Article", "Corpus Evaluation"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SINGLE ARTICLE MODE
    # ════════════════════════════════════════════════════════════════════════
    if summ_mode == "Single Article":

        col_inp2, col_ex2 = st.columns([3, 1])

        with col_ex2:
            st.markdown('<div class="section-label">Quick examples</div>', unsafe_allow_html=True)
            ex_choice2 = st.radio("", list(SUMM_EXAMPLES.keys()), label_visibility="collapsed")
            if st.button("Load example", key="summ_load"):
                st.session_state["summ_article"]   = SUMM_EXAMPLES[ex_choice2]["article"]
                st.session_state["summ_reference"] = SUMM_EXAMPLES[ex_choice2]["reference"]

        with col_inp2:
            article   = st.text_area("Article text",                  key="summ_article",   height=130,
                                      placeholder="Paste the article to summarize…")
            reference = st.text_area("Reference summary (for ROUGE)", key="summ_reference", height=60,
                                      placeholder="A human-written reference summary…")

        if st.button("▶  Summarize & evaluate", type="primary", key="summ_run"):
            if not article:
                st.warning("Please paste an article.")
            else:
                with st.spinner("Running summarization pipeline…"):
                    summary = run_summarization(article)

                st.markdown("##### Generated summary")
                st.markdown(
                    f'<div style="background:#f1efe8;border-radius:8px;padding:10px 14px;'
                    f'font-size:14px;line-height:1.75">{summary}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Abstractive — the model may produce words not present in the article. "
                    "Generation: `do_sample=False`, `num_beams=4`, `no_repeat_ngram_size=3`."
                )

                if reference:
                    st.markdown("##### ROUGE scores (vs. reference)")
                    scores = rouge_scores(predicted=summary, reference=reference)

                    for label, key in [("ROUGE-1", "rouge1"), ("ROUGE-2", "rouge2"), ("ROUGE-L", "rougeL")]:
                        pct = int(scores[key]["fmeasure"] * 100)
                        st.markdown(
                            f'<div class="rouge-row">'
                            f'<span class="rouge-label">{label}</span>'
                            f'<div class="rouge-track"><div class="rouge-fill" style="width:{pct}%"></div></div>'
                            f'<span class="rouge-val">{pct}%</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    col1, col2, col3, col4 = st.columns(4)
                    for col, lbl, key, sub in [
                        (col1, "ROUGE-1 F1", "rouge1", "unigram"),
                        (col2, "ROUGE-2 F1", "rouge2", "bigram"),
                        (col3, "ROUGE-L F1", "rougeL", "LCS"),
                    ]:
                        with col:
                            col.markdown(
                                f'<div class="metric-card">'
                                f'<div class="metric-label">{lbl}</div>'
                                f'<div class="metric-value">{scores[key]["fmeasure"]:.3f}</div>'
                                f'<div class="metric-sub">{sub}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                    with col4:
                        col4.markdown(
                            f'<div class="metric-card">'
                            f'<div class="metric-label">Summary tokens</div>'
                            f'<div class="metric-value">{len(summary.split())}</div>'
                            f'<div class="metric-sub">approx</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    st.caption(
                        "ROUGE-1: unigram F1 · ROUGE-2: bigram F1 · ROUGE-L: LCS F1 · "
                        "`use_stemmer=True` · `scorer.score(reference, predicted)` — reference first."
                    )
                else:
                    st.info("Provide a reference summary to compute ROUGE scores.")

                st.markdown("##### Faithfulness check")
                st.caption(
                    "Numeric values extracted from the article are checked against the summary. "
                    "ROUGE cannot detect factual errors — this check catches what ROUGE misses."
                )

                nums = extract_numbers(article)
                if nums:
                    found_count = 0
                    html = ""
                    for n in nums:
                        found = n in summary
                        if found:
                            found_count += 1
                        cls = "faith-found" if found else "faith-missing"
                        msg = "present in summary" if found else "not found in summary"
                        html += (
                            f'<span class="{cls}">{"✓" if found else "✗"} {n}</span> '
                            f'<span style="font-size:12px;color:#5f5e5a">{msg}</span><br>'
                        )
                    st.markdown(html, unsafe_allow_html=True)

                    ratio = found_count / len(nums)
                    if ratio == 1:
                        st.success(f"All {len(nums)} numeric values from the article appear in the summary.")
                    elif ratio >= 0.5:
                        st.warning(f"{found_count} of {len(nums)} numbers preserved. Review omitted values.")
                    else:
                        st.error(f"Only {found_count} of {len(nums)} numbers present. Summary may omit key quantitative claims.")
                else:
                    st.caption("No numeric values found in the article to check.")

    # ════════════════════════════════════════════════════════════════════════
    # CORPUS EVALUATION MODE
    # ════════════════════════════════════════════════════════════════════════
    else:
        st.markdown("##### Upload a CSV to evaluate")
        st.caption(
            "Required columns: `article_id`, `text`, `reference_summary`. "
            "The pipeline will summarize each article and compute ROUGE against the reference."
        )

        uploaded_summ = st.file_uploader("Choose a CSV file", type=["csv"], key="summ_batch_upload")

        if uploaded_summ is not None:
            try:
                summ_df = pd.read_csv(uploaded_summ)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                summ_df = None

            if summ_df is not None:
                st.markdown(f"**{len(summ_df)} articles loaded** — preview:")
                st.dataframe(summ_df.head(5), use_container_width=True)

                max_rows_summ = st.slider(
                    "Articles to evaluate (reduce for a quick test)",
                    min_value=1,
                    max_value=len(summ_df),
                    value=min(10, len(summ_df)),
                    key="summ_batch_limit",
                )

                if st.button("▶  Run corpus evaluation", type="primary", key="summ_batch_run"):
                    df_eval_summ  = summ_df.head(max_rows_summ).reset_index(drop=True)
                    progress_bar2 = st.progress(0)
                    status_text2  = st.empty()

                    def update_summ_progress(current, total):
                        progress_bar2.progress(current / total)
                        status_text2.caption(f"Summarizing article {current} / {total}…")

                    with st.spinner("Running corpus summarization…"):
                        try:
                            summ_results = evaluate_summaries(
                                df_eval_summ,
                                progress_callback=update_summ_progress,
                            )
                        except ValueError as e:
                            st.error(str(e))
                            summ_results = None

                    progress_bar2.empty()
                    status_text2.empty()

                    if summ_results:
                        st.markdown("##### Aggregate ROUGE scores")
                        c1, c2, c3, c4 = st.columns(4)
                        for col, lbl, val, sub in [
                            (c1, "ROUGE-1", f'{summ_results["rouge1"]:.3f}', "unigram F1"),
                            (c2, "ROUGE-2", f'{summ_results["rouge2"]:.3f}', "bigram F1"),
                            (c3, "ROUGE-L", f'{summ_results["rougeL"]:.3f}', "LCS F1"),
                            (c4, "Articles", str(summ_results["n"]),         "evaluated"),
                        ]:
                            with col:
                                col.markdown(
                                    f'<div class="summ-metric-card">'
                                    f'<div class="metric-label">{lbl}</div>'
                                    f'<div class="metric-value">{val}</div>'
                                    f'<div class="metric-sub">{sub}</div>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                        if summ_results["rouge1"] >= 0.4:
                            st.success("ROUGE-1 ≥ 0.40 — strong lexical overlap with references.")
                        elif summ_results["rouge1"] >= 0.25:
                            st.warning("ROUGE-1 between 0.25–0.40 — moderate overlap. Consider fine-tuning for domain-specific content.")
                        else:
                            st.error("ROUGE-1 < 0.25 — low overlap. The model may not suit this domain without fine-tuning.")

                        st.markdown("##### Per-article predictions")
                        pred_summ_df = pd.DataFrame(summ_results["predictions"])
                        st.dataframe(pred_summ_df, use_container_width=True)

                        st.download_button(
                            label="⬇  Download predictions CSV",
                            data=pred_summ_df.to_csv(index=False).encode("utf-8"),
                            file_name="summary_predictions.csv",
                            mime="text/csv",
                            key="summ_batch_download",
                        )

                        st.caption(
                            "Methodology: ROUGE F1 · `use_stemmer=True` · "
                            "`scorer.score(reference, predicted)` — reference first. "
                            "Model: `sshleifer/distilbart-cnn-6-6`."
                        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — DECISION MATRIX
# ════════════════════════════════════════════════════════════════════════════
with tab_decision:
    st.markdown("#### Five-factor decision matrix")
    st.caption(
        "Adjust the five factors from Section 8 of the reading. "
        "The recommendation updates in real time."
    )

    vals = []
    for i, factor in enumerate(DECISION_FACTORS):
        st.markdown(f"**{factor['name']}**")
        st.caption(f"{factor['desc']} · {factor['lo']} → {factor['hi']}")
        v = st.slider("", 1, 5, 3, key=f"slider_{i}", label_visibility="collapsed")
        vals.append(v)

    weights = [f["weight"] for f in DECISION_FACTORS]
    score   = decision_score(vals, weights)

    st.markdown("---")
    st.markdown(f"**Fine-tune signal score: {score} / 100**")
    st.progress(score / 100)

    rationale = decision_rationale(vals)

    if score < 35:
        st.markdown(
            f'<div class="rec-pretrained">'
            f'<b style="color:#0f6e56">✓ Use pre-trained inference</b><br>'
            f'<span style="font-size:13px;color:#444">{rationale}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif score < 60:
        st.markdown(
            f'<div class="rec-borderline">'
            f'<b style="color:#854f0b">⚖ Evaluate the pre-trained baseline first</b><br>'
            f'<span style="font-size:13px;color:#444">{rationale}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="rec-finetune">'
            f'<b style="color:#534ab7">🧠 Fine-tuning is justified</b><br>'
            f'<span style="font-size:13px;color:#444">{rationale}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**How the score is computed**")
    st.caption(
        "Each factor is weighted and normalized to 0–100. "
        "Weights: labeled data 25%, task specificity 20%, compute budget 20%, "
        "quality gap 20%, iteration speed 15%. "
        "Thresholds: < 35 → pre-trained · 35–60 → evaluate first · > 60 → fine-tune."
    )
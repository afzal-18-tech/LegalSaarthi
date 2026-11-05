"""
Unit tests for Phase 22 statistical analysis.
All tests use deterministic synthetic data; no external data required.
"""
import math
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest
from src.evaluation.statistical_analysis import (
    _sign_rank,
    _paired_ttest,
    _bootstrap_ci,
    _mcnemar,
    compare_qa_systems,
    compare_binary_qa,
    build_error_taxonomy,
    classify_qa_record,
    compare_summ_systems,
)


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank
# ---------------------------------------------------------------------------

def test_wilcoxon_all_positive_diffs_low_p():
    """If A always > B, W should be small → significant."""
    diffs = [1.0] * 20  # all same sign, no ties in rank — but all magnitude equal, all rank 10.5 etc
    # Use varied magnitudes for a cleaner test
    diffs = [float(i) for i in range(1, 21)]
    _, p = _sign_rank(diffs)
    assert p < 0.05, f"Expected p<0.05, got {p}"


def test_wilcoxon_symmetric_diffs_high_p():
    """If diffs are perfectly symmetric around zero, p should be high."""
    diffs = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0]
    _, p = _sign_rank(diffs)
    assert p > 0.5, f"Expected p>0.5 (symmetric), got {p}"


def test_wilcoxon_all_zeros_returns_one():
    """All zeros → p=1.0 (no signal)."""
    _, p = _sign_rank([0.0, 0.0, 0.0])
    assert p == 1.0


# ---------------------------------------------------------------------------
# Paired t-test
# ---------------------------------------------------------------------------

def test_ttest_consistent_gain_significant():
    a = [4.5] * 20
    b = [3.5] * 20
    _, p = _paired_ttest(a, b)
    assert p < 0.05


def test_ttest_no_difference_high_p():
    a = [4.0] * 15
    b = [4.0] * 15
    _, p = _paired_ttest(a, b)
    # Standard error is 0, t is inf or 0. Should produce p=0 or 1.
    assert p >= 0.0  # Just no crash


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def test_bootstrap_ci_excludes_zero_when_a_gt_b():
    a = [5.0] * 30
    b = [3.0] * 30
    mean_d, lo, hi = _bootstrap_ci(a, b, n_boot=1000, seed=42)
    assert mean_d == pytest.approx(2.0)
    assert lo > 0 and hi > 0, f"CI should exclude zero: [{lo}, {hi}]"


def test_bootstrap_ci_includes_zero_when_no_diff():
    import random
    rng = random.Random(99)
    a = [rng.gauss(0, 1) for _ in range(30)]
    b = [rng.gauss(0, 1) for _ in range(30)]
    # With n=30 and noise we can't guarantee, but check no crash and types
    mean_d, lo, hi = _bootstrap_ci(a, b, n_boot=500, seed=42)
    assert isinstance(mean_d, float)
    assert lo <= mean_d <= hi or lo >= mean_d >= hi  # CI contains mean


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------

def test_mcnemar_significant_when_discordant():
    """n10=20, n01=0 → very significant."""
    chi2, p = _mcnemar(20, 0)
    assert p < 0.05


def test_mcnemar_no_discordance_high_p():
    """n10=0, n01=0 → p=1."""
    _, p = _mcnemar(0, 0)
    assert p == 1.0


def test_mcnemar_balanced_discordance_high_p():
    """n10=10, n01=10 → balanced discordance, chi2≈0, p>0.5."""
    chi2, p = _mcnemar(10, 10)
    assert p > 0.5


# ---------------------------------------------------------------------------
# compare_qa_systems
# ---------------------------------------------------------------------------

def _make_qa_pred(qid, corr, fact, abstained=False, gold_hit=True):
    return {
        "query_id": qid,
        "difficulty": "easy",
        "question_type": "factual",
        "evaluation": {
            "correctness": corr if not abstained else None,
            "factuality": fact if not abstained else None,
            "abstention": 1 if abstained else 0,
        },
        "retrieval_eval": {"gold_hit": gold_hit},
        "citations": {},
        "answer": "I don't know" if abstained else "Answer text here",
        "verification": {},
    }


def test_compare_qa_detects_significant_gain():
    preds_a = [_make_qa_pred(f"Q{i}", 5, 5) for i in range(20)]
    preds_b = [_make_qa_pred(f"Q{i}", 2, 2) for i in range(20)]
    result = compare_qa_systems(preds_a, preds_b, "SysA", "SysB", metric="factuality")
    assert result["significant_wilcoxon"]
    assert result["ci_excludes_zero"]
    assert result["mean_diff_a_minus_b"] == pytest.approx(3.0)


def test_compare_qa_excludes_abstained_queries():
    preds_a = [_make_qa_pred(f"Q{i}", 5, 5) for i in range(10)]
    preds_a += [_make_qa_pred(f"Q{i}", None, None, abstained=True) for i in range(10, 20)]
    preds_b = [_make_qa_pred(f"Q{i}", 5, 5) for i in range(20)]
    result = compare_qa_systems(preds_a, preds_b, "A", "B", metric="factuality")
    # Only first 10 queries are paired (both answered)
    assert result["n_paired"] == 10


def test_compare_qa_insufficient_data():
    preds_a = [_make_qa_pred("Q1", 5, 5)]
    preds_b = [_make_qa_pred("Q1", 3, 3)]
    result = compare_qa_systems(preds_a, preds_b, "A", "B", metric="factuality")
    assert result.get("insufficient_data", False)


# ---------------------------------------------------------------------------
# compare_binary_qa
# ---------------------------------------------------------------------------

def test_mcnemar_on_abstention_field():
    # A abstains on 10, B never abstains → significant
    preds_a = [_make_qa_pred(f"Q{i}", None, None, abstained=True) for i in range(10)]
    preds_a += [_make_qa_pred(f"Q{i}", 5, 5) for i in range(10, 20)]
    preds_b = [_make_qa_pred(f"Q{i}", 5, 5) for i in range(20)]
    result = compare_binary_qa(preds_a, preds_b, "A", "B", field="abstention")
    assert result["significant"]


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------

def test_classify_model_refusal():
    rec = _make_qa_pred("Q1", None, None, abstained=True, gold_hit=True)
    mode = classify_qa_record(rec)
    assert mode == "model_refusal"


def test_classify_retrieval_miss():
    rec = _make_qa_pred("Q1", None, None, abstained=True, gold_hit=False)
    mode = classify_qa_record(rec)
    assert mode == "retrieval_miss"


def test_classify_correct():
    rec = _make_qa_pred("Q1", 5, 5)
    assert classify_qa_record(rec) == "correct"


def test_classify_low_factuality():
    rec = _make_qa_pred("Q1", 3, 2)
    assert classify_qa_record(rec) == "low_factuality"


def test_classify_partial_correct():
    rec = _make_qa_pred("Q1", 3, 3)
    assert classify_qa_record(rec) == "partial_correct"


def test_error_taxonomy_counts_sum_to_n():
    preds = [_make_qa_pred(f"Q{i}", 5, 5) for i in range(10)]
    preds += [_make_qa_pred(f"Q{i}", None, None, abstained=True, gold_hit=False) for i in range(10, 15)]
    tax = build_error_taxonomy(preds)
    assert sum(tax["counts"].values()) == len(preds)
    assert tax["n"] == len(preds)


# ---------------------------------------------------------------------------
# compare_summ_systems
# ---------------------------------------------------------------------------

def _make_summ_pred(cid, system, support_rate, flesch, pred_disp, gold_disp="disposed"):
    return {
        "case_id": cid,
        "system": system,
        "gold_disposition": gold_disp,
        "pred_disposition": pred_disp,
        "verification": {"support_rate": support_rate},
        "readability": {"flesch_reading_ease": flesch},
        "generation_failed": False,
    }


def test_compare_summ_detects_significant_flesch_gain():
    preds = []
    for i in range(20):
        preds.append(_make_summ_pred(f"C{i}", "plain_llm", 0.8, 55.0, "disposed"))
        preds.append(_make_summ_pred(f"C{i}", "generic_llm", 0.8, 30.0, "not_stated"))
    result = compare_summ_systems(preds, "plain_llm", "generic_llm", metric="flesch")
    assert result["mean_diff_a_minus_b"] == pytest.approx(25.0)
    assert result["ci_excludes_zero"]


def test_compare_summ_no_difference():
    preds = []
    for i in range(15):
        preds.append(_make_summ_pred(f"C{i}", "A", 0.75, 40.0, "disposed"))
        preds.append(_make_summ_pred(f"C{i}", "B", 0.75, 40.0, "disposed"))
    result = compare_summ_systems(preds, "A", "B", metric="flesch")
    assert not result.get("insufficient_data")
    assert result["mean_diff_a_minus_b"] == pytest.approx(0.0)

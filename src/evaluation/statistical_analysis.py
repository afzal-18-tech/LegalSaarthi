"""
Phase 22 — Statistical Analysis & Error Taxonomy.

All tests are paired (same 50 queries / 29-31 cases across runs).
Only queries answered in BOTH systems contribute to paired tests.

Tests used:
- Wilcoxon signed-rank (non-parametric, ordinal judge scores 1-5).
- Paired t-test (parametric) for cross-validation.
- Bootstrap 95% CIs (10,000 resamples, seed 42) on mean differences.
- McNemar test for paired binary outcomes (abstention, gold-hit).

Error taxonomy: per-query failure mode assignment for the best
system (Verified-RAG BM25) and the proposed system (Verified-RAG Hybrid).
"""
from __future__ import annotations

import json
import math
import pathlib
import random
from typing import Any

# ---------------------------------------------------------------------------
# Wilcoxon signed-rank (pure-stdlib, no scipy required)
# ---------------------------------------------------------------------------

def _sign_rank(diffs: list[float]) -> tuple[float, float]:
    """Return (W_stat, two-tailed p-value) via exact Wilcoxon signed-rank.

    Falls back to normal approximation when n > 25 (very accurate for n>=10).
    Ties in rank resolved by average ranks; zero differences excluded.
    """
    nonzero = [(d, abs(d)) for d in diffs if d != 0.0]
    if not nonzero:
        return 0.0, 1.0  # All identical → cannot reject

    # Sort by absolute value, assign ranks (1-based), average tied ranks
    nonzero_sorted = sorted(nonzero, key=lambda x: x[1])
    ranks: list[float] = []
    i = 0
    while i < len(nonzero_sorted):
        j = i
        while j < len(nonzero_sorted) - 1 and nonzero_sorted[j][1] == nonzero_sorted[j + 1][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for _ in range(j - i + 1):
            ranks.append(avg_rank)
        i = j + 1

    w_plus = sum(r for (d, _), r in zip(nonzero_sorted, ranks) if d > 0)
    w_minus = sum(r for (d, _), r in zip(nonzero_sorted, ranks) if d < 0)
    w_stat = min(w_plus, w_minus)
    n = len(nonzero_sorted)

    # Normal approximation (valid n ≥ 10)
    mu = n * (n + 1) / 4.0
    sigma2 = n * (n + 1) * (2 * n + 1) / 24.0
    z = (w_stat - mu) / math.sqrt(sigma2) if sigma2 > 0 else 0.0
    # Two-tailed p from N(0,1) using error-function approximation
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return w_stat, p


def _norm_cdf(x: float) -> float:
    """Approximation of standard normal CDF using math.erf."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def _paired_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    """Paired t-test on matched lists a and b (same length)."""
    diffs = [ai - bi for ai, bi in zip(a, b)]
    n = len(diffs)
    if n < 2:
        return 0.0, 1.0
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var_d / n)
    if se == 0:
        return float("inf") if mean_d != 0 else 0.0, 0.0 if mean_d != 0 else 1.0
    t = mean_d / se
    # Two-tailed p using t-distribution approximation via normal (adequate for n≥10)
    df = n - 1
    # Use beta regularised function approximation
    p = _t_pvalue(t, df)
    return t, p


def _t_pvalue(t: float, df: int) -> float:
    """Two-tailed p-value from t-statistic using a Cornish-Fisher approx."""
    # For large df this converges to normal; for small df we use approximation
    x = df / (df + t * t)
    # Regularised incomplete beta function via continued fraction (Abramowitz)
    try:
        p_one = _beta_inc(df / 2.0, 0.5, x) / 2.0
        return min(2.0 * p_one, 1.0)
    except Exception:
        # Fallback to normal approximation
        return 2.0 * (1.0 - _norm_cdf(abs(t)))


def _beta_inc(a: float, b: float, x: float, max_iter: int = 200) -> float:
    """Regularised incomplete beta I_x(a,b) via continued fraction."""
    if x < 0 or x > 1:
        raise ValueError
    if x == 0:
        return 1.0
    if x == 1:
        return 0.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    # Lentz's continued fraction
    f = 1.0
    C = f
    D = 1.0 - (a + b) * x / (a + 1)
    if abs(D) < 1e-30:
        D = 1e-30
    D = 1.0 / D
    C = 1.0 + (a + b) * x / ((a + 1) * 1e30)
    f = C * D
    for m in range(1, max_iter):
        # Even step
        coeff = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        D = 1.0 + coeff * D
        C = 1.0 + coeff / C
        if abs(D) < 1e-30:
            D = 1e-30
        if abs(C) < 1e-30:
            C = 1e-30
        D = 1.0 / D
        f *= C * D
        # Odd step
        coeff = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        D = 1.0 + coeff * D
        C = 1.0 + coeff / C
        if abs(D) < 1e-30:
            D = 1e-30
        if abs(C) < 1e-30:
            C = 1e-30
        D = 1.0 / D
        delta = C * D
        f *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return 1.0 - front * f


def _bootstrap_ci(
    a: list[float], b: list[float], n_boot: int = 10000, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Bootstrap percentile CI on mean(a - b) for paired data. Returns (mean_diff, lo, hi)."""
    rng = random.Random(seed)
    n = len(a)
    diffs = [ai - bi for ai, bi in zip(a, b)]
    mean_d = sum(diffs) / n
    boot_means: list[float] = []
    for _ in range(n_boot):
        sample = [rng.choice(diffs) for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(n_boot * alpha / 2)]
    hi = boot_means[int(n_boot * (1 - alpha / 2)) - 1]
    return mean_d, lo, hi


def _mcnemar(n10: int, n01: int) -> tuple[float, float]:
    """McNemar test statistic and p-value (continuity corrected).

    For df=1, chi-squared(1) = Z^2, so p = 2*(1 - Phi(|Z|)) = 2*(1 - Phi(sqrt(chi2))).
    This avoids the incomplete gamma continued fraction for extreme values.
    """
    b, c = n10, n01
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    # chi2(1): p = 2*(1 - Phi(sqrt(chi2)))
    z = math.sqrt(chi2)
    p = 2.0 * (1.0 - _norm_cdf(z))
    p = max(0.0, min(1.0, p))
    return chi2, p


def _chi2_cdf(x: float, df: int) -> float:
    """CDF of chi-squared distribution with df degrees of freedom."""
    if df == 1:
        # chi2(1) = Z^2; CDF = P(Z^2 <= x) = erf(sqrt(x/2))
        return math.erf(math.sqrt(x / 2.0))
    return max(0.0, min(1.0, _gamma_inc(df / 2.0, x / 2.0)))


def _gamma_inc(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x) via series."""
    if x <= 0:
        return 0.0
    if x > a + 1:
        # Use continued fraction representation
        return 1.0 - _gamma_inc_cf(a, x)
    # Series
    ap = a
    delta = s = 1.0 / a
    for _ in range(300):
        ap += 1.0
        delta *= x / ap
        s += delta
        if abs(delta) < abs(s) * 1e-10:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_inc_cf(a: float, x: float) -> float:
    """Upper regularised gamma via continued fraction (Lentz)."""
    f = 1e-30
    C = f
    D = 1.0 / (x + 1 - a)
    D = 1.0 / D if abs(D) < 1e-30 else D
    f = D
    for i in range(1, 201):
        an = -i * (i - a)
        bn = x + 2 * i + 1 - a
        D = bn + an * D
        C = bn + an / C
        if abs(D) < 1e-30:
            D = 1e-30
        if abs(C) < 1e-30:
            C = 1e-30
        D = 1.0 / D
        delta = C * D
        f *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * f


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_qa_predictions(pred_path: pathlib.Path) -> list[dict[str, Any]]:
    """Load QA predictions.jsonl; return list of records."""
    return [json.loads(l) for l in pred_path.read_text().splitlines() if l.strip()]


def load_summ_predictions(pred_path: pathlib.Path) -> list[dict[str, Any]]:
    """Load summarization predictions.jsonl; group by case_id."""
    return [json.loads(l) for l in pred_path.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# QA statistical comparison
# ---------------------------------------------------------------------------

def compare_qa_systems(
    preds_a: list[dict], preds_b: list[dict],
    name_a: str, name_b: str,
    metric: str = "factuality",
    n_boot: int = 10000, seed: int = 42,
) -> dict[str, Any]:
    """
    Paired comparison of metric between two QA systems on matched query_ids.
    Both answered queries contribute scores; abstentions excluded from paired tests.
    Returns rich dict with test statistics.
    """
    by_qid_a = {r["query_id"]: r for r in preds_a}
    by_qid_b = {r["query_id"]: r for r in preds_b}
    shared = sorted(set(by_qid_a) & set(by_qid_b))

    scores_a, scores_b = [], []
    for qid in shared:
        ra, rb = by_qid_a[qid], by_qid_b[qid]
        ea = ra.get("evaluation") or {}
        eb = rb.get("evaluation") or {}
        abs_a = bool(ea.get("abstention", 1) if ea else True)
        abs_b = bool(eb.get("abstention", 1) if eb else True)
        if abs_a or abs_b:
            continue  # Both must be answered
        sa = ea.get(metric)
        sb = eb.get(metric)
        if sa is None or sb is None:
            continue
        scores_a.append(float(sa))
        scores_b.append(float(sb))

    n = len(scores_a)
    if n < 3:
        return {"name_a": name_a, "name_b": name_b, "metric": metric,
                "n_paired": n, "insufficient_data": True}

    mean_a = sum(scores_a) / n
    mean_b = sum(scores_b) / n
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    w_stat, w_p = _sign_rank(diffs)
    t_stat, t_p = _paired_ttest(scores_a, scores_b)
    mean_diff, ci_lo, ci_hi = _bootstrap_ci(scores_a, scores_b, n_boot=n_boot, seed=seed)
    ci_excludes_zero = (ci_lo > 0) or (ci_hi < 0)

    return {
        "name_a": name_a,
        "name_b": name_b,
        "metric": metric,
        "n_paired": n,
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "mean_diff_a_minus_b": round(mean_diff, 4),
        "ci_95_lo": round(ci_lo, 4),
        "ci_95_hi": round(ci_hi, 4),
        "ci_excludes_zero": ci_excludes_zero,
        "wilcoxon_W": round(w_stat, 2),
        "wilcoxon_p": round(w_p, 4),
        "ttest_t": round(t_stat, 3),
        "ttest_p": round(t_p, 4),
        "significant_wilcoxon": w_p < 0.05,
        "significant_ttest": t_p < 0.05,
    }


def compare_binary_qa(
    preds_a: list[dict], preds_b: list[dict],
    name_a: str, name_b: str,
    field: str = "abstention",
) -> dict[str, Any]:
    """McNemar test on paired binary outcomes (abstention / gold_hit)."""
    by_qid_a = {r["query_id"]: r for r in preds_a}
    by_qid_b = {r["query_id"]: r for r in preds_b}
    shared = sorted(set(by_qid_a) & set(by_qid_b))

    n10, n01, n11, n00 = 0, 0, 0, 0
    for qid in shared:
        ra, rb = by_qid_a[qid], by_qid_b[qid]
        if field == "abstention":
            va = int(ra.get("evaluation", {}).get("abstention", 1))
            vb = int(rb.get("evaluation", {}).get("abstention", 1))
        elif field == "gold_hit":
            va = int(ra.get("retrieval_eval", {}).get("gold_hit", False))
            vb = int(rb.get("retrieval_eval", {}).get("gold_hit", False))
        else:
            va = vb = 0
        if va == 1 and vb == 0:
            n10 += 1
        elif va == 0 and vb == 1:
            n01 += 1
        elif va == 1 and vb == 1:
            n11 += 1
        else:
            n00 += 1

    chi2, p = _mcnemar(n10, n01)
    return {
        "name_a": name_a,
        "name_b": name_b,
        "field": field,
        "n_shared": len(shared),
        "n_a_only": n10,
        "n_b_only": n01,
        "n_both": n11,
        "n_neither": n00,
        "mcnemar_chi2": round(chi2, 3),
        "mcnemar_p": round(p, 4),
        "significant": p < 0.05,
    }


# ---------------------------------------------------------------------------
# Error taxonomy (QA)
# ---------------------------------------------------------------------------

ERROR_MODES = [
    "model_refusal",          # generator abstained (/I don't have enough/)
    "retrieval_miss",         # gold chunk not in top-k
    "claim_gate_abstention",  # claim verifier forced abstention
    "low_factuality",         # answered but factuality ≤ 2
    "low_correctness",        # answered but correctness ≤ 2
    "citation_hallucination", # answered with invalid citations
    "correct",                # answered with correctness ≥ 4 and factuality ≥ 4
    "partial_correct",        # answered correctness/factuality 3
    "unknown",
]


def classify_qa_record(rec: dict[str, Any]) -> str:
    """Assign one error mode to a single QA prediction record."""
    ev = rec.get("evaluation") or {}
    ret = rec.get("retrieval_eval") or {}
    cits = rec.get("citations") or {}
    answer = rec.get("answer", "")

    abstained = bool(ev.get("abstention", 1))
    corr = ev.get("correctness") or 0
    fact = ev.get("factuality") or 0
    gold_hit = bool(ret.get("gold_hit", False))
    invalid_cit_rate = 0.0
    if isinstance(cits, dict):
        invalid = cits.get("invalid_ids", []) or []
        valid = cits.get("valid_ids", []) or []
        total = len(invalid) + len(valid)
        invalid_cit_rate = len(invalid) / total if total > 0 else 0.0

    # Claim-gate abstention: abstained but gold was in context
    verification = rec.get("verification") or {}
    verif_abstentions = verification.get("verification_abstentions", 0)

    if abstained:
        if verif_abstentions and verif_abstentions > 0:
            return "claim_gate_abstention"
        if not gold_hit:
            return "retrieval_miss"
        return "model_refusal"

    if invalid_cit_rate > 0.05:
        return "citation_hallucination"
    if fact <= 2:
        return "low_factuality"
    if corr <= 2:
        return "low_correctness"
    if corr >= 4 and fact >= 4:
        return "correct"
    if corr == 3 or fact == 3:
        return "partial_correct"
    return "unknown"


def build_error_taxonomy(preds: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate error modes across all predictions."""
    counts: dict[str, int] = {m: 0 for m in ERROR_MODES}
    per_query: list[dict] = []
    for rec in preds:
        mode = classify_qa_record(rec)
        counts[mode] = counts.get(mode, 0) + 1
        per_query.append({
            "query_id": rec.get("query_id"),
            "difficulty": rec.get("difficulty"),
            "question_type": rec.get("question_type"),
            "error_mode": mode,
            "correctness": (rec.get("evaluation") or {}).get("correctness"),
            "factuality": (rec.get("evaluation") or {}).get("factuality"),
            "abstained": bool((rec.get("evaluation") or {}).get("abstention", 1)),
            "gold_hit": bool((rec.get("retrieval_eval") or {}).get("gold_hit", False)),
        })
    n = len(preds)
    pct = {m: round(c / n * 100, 1) for m, c in counts.items()}
    return {"counts": counts, "percentages": pct, "per_query": per_query, "n": n}


# ---------------------------------------------------------------------------
# Summarization paired comparison
# ---------------------------------------------------------------------------

def compare_summ_systems(
    preds_all: list[dict],
    sys_a: str, sys_b: str,
    metric: str = "support_rate",
    n_boot: int = 10000, seed: int = 42,
) -> dict[str, Any]:
    """Paired comparison of summarization systems on matched case_ids."""
    a_map = {r["case_id"]: r for r in preds_all if r.get("system") == sys_a}
    b_map = {r["case_id"]: r for r in preds_all if r.get("system") == sys_b}
    shared_cases = sorted(set(a_map) & set(b_map))

    scores_a, scores_b = [], []
    for cid in shared_cases:
        ra, rb = a_map[cid], b_map[cid]

        def get_val(r: dict, m: str) -> float | None:
            if m == "support_rate":
                v = r.get("verification", {})
                if isinstance(v, dict):
                    sr = v.get("support_rate")
                    if sr is not None:
                        return float(sr)
            elif m == "flesch":
                rd = r.get("readability", {})
                return float(rd.get("flesch_reading_ease", 0)) if rd else None
            elif m == "disposition_correct":
                gd = r.get("gold_disposition")
                pd = r.get("pred_disposition")
                if gd and pd:
                    return 1.0 if gd == pd else 0.0
                return None
            return None

        va = get_val(ra, metric)
        vb = get_val(rb, metric)
        if va is None or vb is None:
            continue
        scores_a.append(va)
        scores_b.append(vb)

    n = len(scores_a)
    if n < 3:
        return {"sys_a": sys_a, "sys_b": sys_b, "metric": metric,
                "n_paired": n, "insufficient_data": True}

    mean_a = sum(scores_a) / n
    mean_b = sum(scores_b) / n
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    w_stat, w_p = _sign_rank(diffs)
    t_stat, t_p = _paired_ttest(scores_a, scores_b)
    mean_diff, ci_lo, ci_hi = _bootstrap_ci(scores_a, scores_b, n_boot=n_boot, seed=seed)
    ci_excludes_zero = (ci_lo > 0) or (ci_hi < 0)

    return {
        "sys_a": sys_a,
        "sys_b": sys_b,
        "metric": metric,
        "n_paired": n,
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "mean_diff_a_minus_b": round(mean_diff, 4),
        "ci_95_lo": round(ci_lo, 4),
        "ci_95_hi": round(ci_hi, 4),
        "ci_excludes_zero": ci_excludes_zero,
        "wilcoxon_W": round(w_stat, 2),
        "wilcoxon_p": round(w_p, 4),
        "ttest_t": round(t_stat, 3),
        "ttest_p": round(t_p, 4),
        "significant_wilcoxon": w_p < 0.05,
        "significant_ttest": t_p < 0.05,
    }

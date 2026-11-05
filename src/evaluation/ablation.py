"""
LegalSaarthi — Phase 21 ablation analysis helpers.

Everything here is a pure function over saved prediction records so ablations that
only change a *post-hoc decision* (claim-gating threshold, citation stripping) can be
derived from raw Phase 20 predictions without re-running the generator.

Honesty notes encoded in the API:
  * Differences are reported with seeded bootstrap CIs; with n<=50 most are not
    statistically established (formal tests are Phase 22).
  * Judge scores exist only for answered queries; paired comparisons use the
    intersection of queries answered (and scored) by both systems.
"""
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.generation.rag_pipeline import extract_citations

BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10000


def _mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def bootstrap_mean_ci(values: Sequence[float], seed: int = BOOTSTRAP_SEED,
                      n_resamples: int = BOOTSTRAP_RESAMPLES) -> Dict[str, Optional[float]]:
    """Percentile bootstrap 95% CI of the mean."""
    vals = list(values)
    if not vals:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    return {"mean": sum(vals) / n, "lo": means[int(0.025 * n_resamples)],
            "hi": means[int(0.975 * n_resamples) - 1], "n": n}


def paired_bootstrap_diff(a: Dict[str, float], b: Dict[str, float], seed: int = BOOTSTRAP_SEED,
                          n_resamples: int = BOOTSTRAP_RESAMPLES) -> Dict[str, Any]:
    """Paired bootstrap CI for mean(b - a) over queries present in both dicts."""
    common = sorted(set(a) & set(b))
    if not common:
        return {"n_paired": 0, "mean_diff": None, "lo": None, "hi": None, "excludes_zero": None}
    diffs = [b[q] - a[q] for q in common]
    ci = bootstrap_mean_ci(diffs, seed=seed, n_resamples=n_resamples)
    return {"n_paired": len(common), "mean_diff": ci["mean"], "lo": ci["lo"], "hi": ci["hi"],
            "excludes_zero": bool(ci["lo"] > 0 or ci["hi"] < 0)}


def scored_vector(rows: List[Dict[str, Any]], field: str, split: Optional[str] = None) -> Dict[str, float]:
    """query_id -> judge score for answered, successfully-judged queries (abstentions excluded)."""
    out: Dict[str, float] = {}
    for r in rows:
        ev = r.get("evaluation", {})
        if split and r.get("split") != split:
            continue
        if r.get("generation_failed") or ev.get("abstention") == 1 or ev.get("judge_parse_failure"):
            continue
        if ev.get(field) is None:
            continue
        out[r["query_id"]] = float(ev[field])
    return out


def abstention_breakdown(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Why did each abstained query abstain? Separates generator refusals from verifier gating.

    Basic-RAG rows have no `verification` block, so every abstention there is a model refusal.
    """
    out = {"total": len(rows), "abstained": 0, "model_refusal": 0, "claim_verifier_gate": 0,
           "other_or_unlabelled": 0}
    for r in rows:
        if r.get("evaluation", {}).get("abstention") != 1:
            continue
        out["abstained"] += 1
        v = r.get("verification")
        if v is None:
            out["model_refusal"] += 1
            continue
        reason = v.get("abstention_reason") or ""
        if reason == "model_abstained":
            out["model_refusal"] += 1
        elif reason.startswith("support_rate"):
            out["claim_verifier_gate"] += 1
        else:
            out["other_or_unlabelled"] += 1
    return out


def citation_ids_in(text: str) -> List[str]:
    return extract_citations(text or "")


def citation_integrity(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Invalid-citation rate before vs after the citation verifier, for Verified-RAG rows.

    before: IDs cited in `raw_answer` that were not retrieved.
    after : IDs cited in the *delivered* `answer` that were not retrieved.
    Only answered queries are counted (abstentions deliver no citations).
    """
    cited_before = invalid_before = cited_after = invalid_after = 0
    for r in rows:
        if r.get("evaluation", {}).get("abstention") == 1 or r.get("generation_failed"):
            continue
        retrieved = {c["chunk_id"] for c in r.get("retrieved", [])}
        raw = citation_ids_in(r.get("raw_answer", r.get("answer", "")))
        delivered = citation_ids_in(r.get("answer", ""))
        cited_before += len(raw)
        invalid_before += sum(1 for c in raw if c not in retrieved)
        cited_after += len(delivered)
        invalid_after += sum(1 for c in delivered if c not in retrieved)
    return {
        "cited_before": cited_before, "invalid_before": invalid_before,
        "invalid_rate_before": invalid_before / cited_before if cited_before else None,
        "cited_after": cited_after, "invalid_after": invalid_after,
        "invalid_rate_after": invalid_after / cited_after if cited_after else None,
    }


def _ranks(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation with tie handling; None if undefined."""
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return None if dx == 0 or dy == 0 else num / (dx * dy)


def support_vs_judged_quality(rows: List[Dict[str, Any]], field: str = "factuality") -> Dict[str, Any]:
    """Does the claim verifier's support_rate track judged quality? (answered queries only)."""
    xs, ys = [], []
    for r in rows:
        ev = r.get("evaluation", {})
        if r.get("generation_failed") or ev.get("abstention") == 1 or ev.get("judge_parse_failure"):
            continue
        if "verification" not in r or ev.get(field) is None:
            continue
        xs.append(float(r["verification"]["support_rate"]))
        ys.append(float(ev[field]))
    return {"n": len(xs), "spearman": spearman(xs, ys)}


def claim_gate_counterfactual(rows: List[Dict[str, Any]], tau: float,
                              raw_judgements: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Re-decide each Verified-RAG query at threshold `tau` without re-generating.

    Model refusals and generation failures stay abstentions at every tau. Otherwise a
    query is answered iff support_rate >= tau. Judge scores for answers that the original
    run withheld come from `raw_judgements` (query_id -> {'correctness','factuality'});
    if a needed judgement is missing the query is counted in `missing_judgement` and
    excluded from quality means (never imputed).
    """
    answered_scores_c: List[float] = []
    answered_scores_f: List[float] = []
    answered = abstained = missing = 0
    for r in rows:
        qid = r["query_id"]
        ev = r.get("evaluation", {})
        v = r.get("verification", {})
        model_refusal = v.get("abstention_reason") == "model_abstained"
        reason = str(v.get("abstention_reason") or "")
        withheld_by_gate = ev.get("abstention") == 1 and reason.startswith("support_rate")
        withheld_other = ev.get("abstention") == 1 and not withheld_by_gate
        if r.get("generation_failed") or model_refusal or withheld_other:
            abstained += 1
            continue
        if float(v.get("support_rate", 0.0)) < tau:
            abstained += 1
            continue
        answered += 1
        if ev.get("abstention") == 0 and not ev.get("judge_parse_failure") and ev.get("correctness") is not None:
            answered_scores_c.append(float(ev["correctness"]))
            answered_scores_f.append(float(ev["factuality"]))
        elif qid in raw_judgements and not raw_judgements[qid].get("judge_parse_failure") \
                and raw_judgements[qid].get("abstention") == 0:
            answered_scores_c.append(float(raw_judgements[qid]["correctness"]))
            answered_scores_f.append(float(raw_judgements[qid]["factuality"]))
        else:
            missing += 1
    total = len(rows)
    return {
        "tau": tau, "total": total, "answered": answered, "abstained": abstained,
        "coverage": answered / total if total else None,
        "scored_answered": len(answered_scores_f), "missing_judgement": missing,
        "correctness_answered": _mean(answered_scores_c), "factuality_answered": _mean(answered_scores_f),
    }


def verifier_gated_queries(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows whose answer was withheld *only* because support_rate < tau (need raw-answer judging)."""
    out = []
    for r in rows:
        v = r.get("verification") or {}
        if r.get("evaluation", {}).get("abstention") == 1 and str(v.get("abstention_reason") or "").startswith(
                "support_rate"):
            out.append(r)
    return out

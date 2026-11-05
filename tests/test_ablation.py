"""Phase 21 ablation helper tests (pure functions; no LLM)."""
from src.evaluation import ablation as A


def _row(qid, abst=0, corr=4, fact=4, support=1.0, reason=None, raw="x [c1]", ans="x [c1]",
         retrieved=("c1", "c2"), gen_failed=False, verified=True, split="val"):
    r = {
        "query_id": qid, "split": split, "generation_failed": gen_failed,
        "evaluation": {"abstention": abst, "correctness": None if abst else corr,
                       "factuality": None if abst else fact, "judge_parse_failure": False},
        "retrieved": [{"chunk_id": c} for c in retrieved],
        "raw_answer": raw, "answer": ans,
    }
    if verified:
        r["verification"] = {"support_rate": support, "abstention_reason": reason}
    return r


def test_bootstrap_ci_contains_mean_and_is_seeded():
    vals = [1, 2, 3, 4, 5]
    a = A.bootstrap_mean_ci(vals, n_resamples=2000)
    b = A.bootstrap_mean_ci(vals, n_resamples=2000)
    assert a == b
    assert a["lo"] <= a["mean"] <= a["hi"] and a["mean"] == 3


def test_paired_diff_detects_consistent_gain_and_ignores_unpaired():
    a = {f"q{i}": 3.0 for i in range(10)}
    b = {f"q{i}": 4.0 for i in range(10)}
    b["only_b"] = 1.0
    d = A.paired_bootstrap_diff(a, b, n_resamples=2000)
    assert d["n_paired"] == 10 and d["mean_diff"] == 1.0 and d["excludes_zero"] is True


def test_paired_diff_no_overlap():
    assert A.paired_bootstrap_diff({"a": 1.0}, {"b": 2.0})["n_paired"] == 0


def test_abstention_breakdown_separates_gate_from_refusal():
    rows = [
        _row("q1", abst=1, reason="model_abstained"),
        _row("q2", abst=1, reason="support_rate=0.20 < threshold=0.50"),
        _row("q3", abst=1, reason=None),
        _row("q4"),
        _row("q5", abst=1, verified=False),
    ]
    b = A.abstention_breakdown(rows)
    assert b["abstained"] == 4
    assert b["model_refusal"] == 2 and b["claim_verifier_gate"] == 1 and b["other_or_unlabelled"] == 1


def test_citation_integrity_before_after():
    rows = [_row("q1", raw="a [c1] b [zzz]", ans="a [c1] b"),
            _row("q2", abst=1, raw="[bad]", ans="Insufficient evidence to answer.")]
    c = A.citation_integrity(rows)
    assert c["cited_before"] == 2 and c["invalid_before"] == 1
    assert c["cited_after"] == 1 and c["invalid_after"] == 0


def test_spearman_perfect_and_undefined():
    assert abs(A.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(A.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    assert A.spearman([1, 1, 1], [1, 2, 3]) is None
    assert A.spearman([1, 2], [1, 2]) is None


def test_claim_gate_counterfactual_keeps_model_refusals_and_uses_overrides():
    rows = [
        _row("q1", support=0.9, corr=5, fact=5),
        _row("q2", abst=1, support=0.2, reason="support_rate=0.20 < threshold=0.50"),
        _row("q3", abst=1, reason="model_abstained", support=1.0),
        _row("q4", support=0.3, corr=2, fact=2),
    ]
    overrides = {"q2": {"correctness": 3, "factuality": 3, "abstention": 0, "judge_parse_failure": False}}
    off = A.claim_gate_counterfactual(rows, tau=0.0, raw_judgements=overrides)
    assert off["answered"] == 3 and off["abstained"] == 1 and off["missing_judgement"] == 0
    assert abs(off["factuality_answered"] - (5 + 3 + 2) / 3) < 1e-9
    on = A.claim_gate_counterfactual(rows, tau=0.5, raw_judgements=overrides)
    assert on["answered"] == 1 and on["abstained"] == 3
    assert on["factuality_answered"] == 5.0


def test_claim_gate_counterfactual_never_imputes_missing_judgement():
    rows = [_row("q2", abst=1, support=0.2, reason="support_rate=0.20 < threshold=0.50")]
    off = A.claim_gate_counterfactual(rows, tau=0.0, raw_judgements={})
    assert off["answered"] == 1 and off["missing_judgement"] == 1
    assert off["factuality_answered"] is None


def test_verifier_gated_queries_selects_only_gate_abstentions():
    rows = [_row("q1", abst=1, reason="support_rate=0.1 < threshold=0.5"),
            _row("q2", abst=1, reason="model_abstained"), _row("q3")]
    assert [r["query_id"] for r in A.verifier_gated_queries(rows)] == ["q1"]


def test_scored_vector_excludes_abstentions_and_split_filter():
    rows = [_row("q1", fact=5, split="val"), _row("q2", abst=1), _row("q3", fact=3, split="test")]
    assert A.scored_vector(rows, "factuality") == {"q1": 5.0, "q3": 3.0}
    assert A.scored_vector(rows, "factuality", split="test") == {"q3": 3.0}

"""Unit tests for Phase 18 robustness perturbations (no LLM required)."""
from src.evaluation.robustness import (
    PerturbingRetriever, summarize_condition, tamper_numbers,
)


class FakeBase:
    def retrieve(self, q, top_k=5):
        return [("g1", 0.9), ("g2", 0.8), ("x1", 0.7)][:top_k]


def store():
    return {
        "g1": {"chunk_id": "g1", "case_id": "A", "text": "Section 302 in 1998 held."},
        "g2": {"chunk_id": "g2", "case_id": "A", "text": "Second gold."},
        "x1": {"chunk_id": "x1", "case_id": "A", "text": "Other."},
        "o1": {"chunk_id": "o1", "case_id": "B", "text": "Foreign."},
        "o2": {"chunk_id": "o2", "case_id": "C", "text": "Foreign2."},
    }


def mk(cond):
    r = PerturbingRetriever(FakeBase(), store(), cond)
    r.set_gold(["g1", "g2"], "A")
    return r


def test_clean_unchanged():
    assert [h[0] for h in mk("clean").retrieve("q")] == ["g1", "g2", "x1"]


def test_no_evidence_removes_gold():
    assert [h[0] for h in mk("no_evidence").retrieve("q")] == ["x1"]


def test_partial_drops_half_of_gold():
    ids = [h[0] for h in mk("partial").retrieve("q")]
    assert "g1" in ids and "g2" not in ids


def test_distractor_other_cases_only_and_deterministic():
    r = mk("distractor")
    a = [h[0] for h in r.retrieve("q")]
    assert set(a) <= {"o1", "o2"} and a == [h[0] for h in r.retrieve("q")]


def test_contradictory_adds_tampered_copy():
    r = mk("contradictory")
    ids = [h[0] for h in r.retrieve("q")]
    assert "g1__tampered" in ids
    assert "305" not in r.store["g1__tampered"]["text"]
    assert "309" in r.store["g1__tampered"]["text"]


def test_partial_single_gold_truncates_and_restores():
    r = PerturbingRetriever(FakeBase(), store(), "partial")
    r.set_gold(["g1"], "A")
    r.retrieve("q")
    assert len(r.store["g1"]["text"]) < len("Section 302 in 1998 held.")
    r.restore()
    assert r.store["g1"]["text"] == "Section 302 in 1998 held."


def test_tamper_numbers():
    assert tamper_numbers("s 10 of 2000") == "s 17 of 2007"


def test_summarize_abstain_metrics():
    rec = lambda ab: {"verification": {"abstained": ab, "support_rate": 0.5},
                      "citations": {"invalid_ids": []}, "evaluation": {}}
    s = summarize_condition([rec(True), rec(True), rec(False)], "no_evidence")
    assert abs(s["correct_abstention_rate"] - 2 / 3) < 1e-9
    assert abs(s["unsupported_answer_rate"] - 1 / 3) < 1e-9

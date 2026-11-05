"""
Unit tests for Phase 23 harness — all synthetic, no LLM calls, no file I/O.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest
from src.evaluation.phase23_harness import (
    AUDIT_QUESTIONS,
    parse_judge_rubric,
    check_citation_validity,
    check_statutory_references,
    check_evidence_traceability,
    is_abstention,
    compute_agreement,
    aggregate_rubric_scores,
)


# ── Audit question list ───────────────────────────────────────────────────────
def test_audit_questions_count():
    assert len(AUDIT_QUESTIONS) == 50

def test_audit_questions_unique_qids():
    qids = [q["qid"] for q in AUDIT_QUESTIONS]
    assert len(set(qids)) == 50

def test_audit_questions_have_required_fields():
    for q in AUDIT_QUESTIONS:
        assert "qid" in q and "category" in q and "prompt" in q

def test_audit_questions_categories():
    cats = {q["category"] for q in AUDIT_QUESTIONS}
    expected = {
        "Facts & background", "Legal issues", "Law & precedents",
        "Reasoning", "Decision & relief", "Summary", "Citation & evidence"
    }
    assert cats == expected


# ── Judge rubric parsing ──────────────────────────────────────────────────────
def test_parse_rubric_all_present():
    raw = (
        "FACTUALITY: 5\n"
        "COMPLETENESS: 4\n"
        "CLARITY: 3\n"
        "LEGAL_COHERENCE: 4\n"
        "CITATION_VALIDITY: 5\n"
        "EVIDENCE_SUPPORT: 4\n"
    )
    scores = parse_judge_rubric(raw)
    assert scores["factuality"] == 5
    assert scores["completeness"] == 4
    assert scores["clarity"] == 3
    assert scores["legal_coherence"] == 4
    assert scores["citation_validity"] == 5
    assert scores["evidence_support"] == 4
    assert not scores["parse_failure"]

def test_parse_rubric_missing_dimension():
    raw = "FACTUALITY: 5\nCOMPLETENESS: 3\nCLARITY: 4\n"
    scores = parse_judge_rubric(raw)
    assert scores["factuality"] == 5
    assert scores["legal_coherence"] is None
    assert scores["parse_failure"]

def test_parse_rubric_case_insensitive():
    raw = "factuality: 3\ncompleteness: 3\nclarity: 3\nlegal_coherence: 3\ncitation_validity: 3\nevidence_support: 3"
    scores = parse_judge_rubric(raw)
    assert not scores["parse_failure"]

def test_parse_rubric_empty_string():
    scores = parse_judge_rubric("")
    assert scores["parse_failure"]


# ── Citation validity ─────────────────────────────────────────────────────────
def test_citation_validity_all_valid():
    result = check_citation_validity(
        answer="See [CHUNK abc] for details.",
        retrieved_chunk_ids=["abc", "def", "ghi"],
        cited_ids=["abc"]
    )
    assert result["n_valid"] == 1
    assert result["n_invalid"] == 0
    assert result["citation_precision"] == 1.0

def test_citation_validity_some_invalid():
    result = check_citation_validity(
        answer="See [CHUNK abc] and [CHUNK xyz]",
        retrieved_chunk_ids=["abc", "def"],
        cited_ids=["abc", "xyz"]
    )
    assert result["n_invalid"] == 1
    assert result["citation_precision"] == 0.5

def test_citation_validity_no_citations():
    result = check_citation_validity(
        answer="The court held ...",
        retrieved_chunk_ids=["abc"],
        cited_ids=[]
    )
    assert result["n_cited"] == 0
    assert result["citation_precision"] == 1.0
    assert not result["has_citation"]


# ── Statutory reference detection ─────────────────────────────────────────────
def test_statutory_detection_section():
    result = check_statutory_references("Under Section 14 of the Act, the court held...")
    assert result["n_statutory"] > 0
    assert result["has_statutory"]

def test_statutory_detection_article():
    result = check_statutory_references("Article 226 of the Constitution empowers High Courts...")
    assert result["has_statutory"]

def test_statutory_detection_none():
    result = check_statutory_references("The court rejected the appeal.")
    assert not result["has_statutory"]
    assert result["n_statutory"] == 0

def test_statutory_detection_case_citation():
    result = check_statutory_references(
        "As held in 2015 INSC 177, the principle applies."
    )
    assert result["n_case_refs"] > 0
    assert result["has_case_refs"]


# ── Evidence traceability ─────────────────────────────────────────────────────
def test_traceability_high_when_answer_from_chunks():
    chunk_text = "The court held that the contract was validly executed on 10th January 2015"
    chunks = [{"chunk_id": "c1", "text": chunk_text}]
    result = check_evidence_traceability(
        "The court held that the contract was validly executed on 10th January 2015",
        chunks
    )
    assert result["traceability_rate"] > 0.8

def test_traceability_zero_without_chunks():
    result = check_evidence_traceability("Some answer text here.", [])
    assert result["traceability_rate"] == 0.0

def test_traceability_low_for_hallucination():
    chunk_text = "The court dismissed the appeal filed by the respondent."
    chunks = [{"chunk_id": "c1", "text": chunk_text}]
    result = check_evidence_traceability(
        "The petitioner was awarded damages of Rs. 50 lakhs by the Supreme Court bench.",
        chunks
    )
    # Very little overlap
    assert result["traceability_rate"] < 0.3


# ── Abstention detection ──────────────────────────────────────────────────────
def test_abstention_detected():
    assert is_abstention("I don't know.")
    assert is_abstention("Insufficient information in the provided excerpts.")
    assert is_abstention("I cannot determine the answer from the given information.")

def test_abstention_not_detected_for_real_answer():
    answer = (
        "The Supreme Court held that Section 14 of the Contract Act applied to the present case. "
        "The appellant's contention was rejected as the Court found sufficient grounds to dismiss "
        "the appeal. The final order directed the appellant to pay costs of Rs. 10,000."
    )
    assert not is_abstention(answer)

def test_abstention_long_answer_not_abstention():
    # Even if it contains "I don't know" in a long answer, should not be abstention
    answer = "I don't know if this is the right interpretation but the court held that... " * 20
    assert not is_abstention(answer)


# ── Agreement analysis ────────────────────────────────────────────────────────
def test_compute_agreement_perfect():
    result = compute_agreement([5, 4, 3, 2, 1], [5, 4, 3, 2, 1])
    assert result["exact_agreement"] == 1.0
    assert result["mae"] == 0.0

def test_compute_agreement_all_wrong():
    result = compute_agreement([1, 1, 1, 1, 1], [5, 5, 5, 5, 5])
    assert result["exact_agreement"] == 0.0
    assert result["mae"] == 4.0

def test_compute_agreement_adjacent():
    result = compute_agreement([4, 4, 4], [5, 5, 5])
    assert result["adjacent_agreement"] == 1.0

def test_compute_agreement_empty():
    result = compute_agreement([], [])
    assert result["n"] == 0


# ── Aggregate rubric scores ───────────────────────────────────────────────────
def _make_rec(fact, comp, clar, lc, cv, es, abstained=False):
    return {
        "judge_scores": {
            "factuality": fact, "completeness": comp, "clarity": clar,
            "legal_coherence": lc, "citation_validity": cv, "evidence_support": es,
            "parse_failure": False,
        },
        "abstained": abstained,
    }

def test_aggregate_scores_means():
    recs = [
        _make_rec(5, 4, 5, 4, 5, 5),
        _make_rec(3, 3, 3, 3, 3, 3),
    ]
    agg = aggregate_rubric_scores(recs)
    assert agg["mean_factuality"] == pytest.approx(4.0)
    assert agg["mean_clarity"] == pytest.approx(4.0)
    assert agg["n"] == 2

def test_aggregate_counts_abstentions():
    recs = [
        _make_rec(3, 1, 3, 3, 3, 3, abstained=True),
        _make_rec(5, 4, 5, 4, 5, 5),
    ]
    agg = aggregate_rubric_scores(recs)
    assert agg["n_abstained"] == 1
    assert agg["abstention_rate"] == pytest.approx(0.5)

def test_aggregate_composite_is_mean_of_all():
    recs = [_make_rec(4, 4, 4, 4, 4, 4)]
    agg = aggregate_rubric_scores(recs)
    assert agg["mean_composite"] == pytest.approx(4.0)

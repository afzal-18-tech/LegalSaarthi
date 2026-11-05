"""
LegalSaarthi — Phase 14 Verified-RAG Pipeline Unit Tests

Tests cover:
 - CitationVerifier: valid/invalid citation detection and stripping
 - ClaimVerifier: claim extraction, verdict parsing
 - VerifiedRAGPipeline: abstention logic, end-to-end flow
 - Verification aggregation in generation_eval
All tests are fully offline (no LLM calls).
"""
import pytest
from unittest.mock import MagicMock

from src.generation.verified_rag_pipeline import (
    CitationVerifier,
    ClaimVerifier,
    VerifiedRAGPipeline,
)
from src.evaluation.generation_eval import aggregate


# ── Stub LLM for offline testing ──────────────────────────────────────────
class StubLLM:
    def __init__(self, response: str):
        self._response = response
        self.last_model_used = "stub"
        self.last_latency_s = 0.01

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        return self._response


class StubRetriever:
    def __init__(self, results):
        self._results = results  # list of (chunk_id, score)

    def retrieve(self, query: str, top_k: int = 10):
        return self._results[:top_k]


# ── CitationVerifier tests ────────────────────────────────────────────────

def test_citation_verifier_all_valid():
    answer = "The court held [chunk_001]. The appeal was dismissed [chunk_002]."
    result = CitationVerifier.verify(answer, retrieved_ids=["chunk_001", "chunk_002"])
    assert result["valid_ids"] == ["chunk_001", "chunk_002"]
    assert result["invalid_ids"] == []
    assert result["citations_stripped"] == 0
    assert result["cleaned_answer"] == answer


def test_citation_verifier_strips_invalid():
    answer = "The court held [REAL_001]. It also cited [FAKE_999]."
    result = CitationVerifier.verify(answer, retrieved_ids=["REAL_001"])
    assert result["valid_ids"] == ["REAL_001"]
    assert result["invalid_ids"] == ["FAKE_999"]
    assert result["citations_stripped"] == 1
    assert "FAKE_999" not in result["cleaned_answer"]
    assert "REAL_001" in result["cleaned_answer"]


def test_citation_verifier_strips_chunk_prefix_and_parentheses():
    answer = "The court held [CHUNK REAL_001]. It also cited (CHUNK FAKE_999)."
    result = CitationVerifier.verify(answer, retrieved_ids=["REAL_001"])
    assert result["valid_ids"] == ["REAL_001"]
    assert result["invalid_ids"] == ["FAKE_999"]
    assert result["citations_stripped"] == 1
    assert "FAKE_999" not in result["cleaned_answer"]
    assert "REAL_001" in result["cleaned_answer"]



def test_citation_verifier_no_citations():
    answer = "The court ruled in favour of the appellant."
    result = CitationVerifier.verify(answer, retrieved_ids=["chunk_001"])
    assert result["cited_ids"] == []
    assert result["has_citation"] is False
    assert result["citations_stripped"] == 0


def test_citation_verifier_all_invalid():
    answer = "See [GHOST_A] and [GHOST_B]."
    result = CitationVerifier.verify(answer, retrieved_ids=["chunk_real"])
    assert result["valid_ids"] == []
    assert result["invalid_ids"] == ["GHOST_A", "GHOST_B"]
    assert result["citations_stripped"] == 2


# ── ClaimVerifier tests ───────────────────────────────────────────────────

def test_claim_extractor_basic():
    answer = "The court dismissed the appeal. The fine was refunded to the appellant."
    claims = ClaimVerifier.extract_claims(answer)
    assert len(claims) == 2
    assert "dismissed the appeal" in claims[0]


def test_claim_extractor_skips_abstention():
    answer = "Insufficient evidence to answer."
    claims = ClaimVerifier.extract_claims(answer)
    assert claims == []


def test_claim_extractor_skips_short_fragments():
    answer = "Yes. The court dismissed the appeal on grounds of insufficient evidence."
    claims = ClaimVerifier.extract_claims(answer)
    # "Yes." is too short (<15 chars), long sentence should be kept
    assert all(len(c) >= 15 for c in claims)


def test_claim_verifier_parses_supported():
    evidence = "The High Court dismissed the appeal dated 26.11.1992."
    claims = ["The High Court dismissed the appeal."]
    stub_llm = StubLLM("CLAIM 1: SUPPORTED")
    verifier = ClaimVerifier(stub_llm)
    result = verifier.verify_claims(claims, evidence)
    assert result["supported_count"] == 1
    assert result["unsupported_count"] == 0
    assert result["support_rate"] == 1.0
    assert result["verdicts"] == ["SUPPORTED"]


def test_claim_verifier_parses_unsupported():
    evidence = "The court upheld the conviction."
    claims = ["The court acquitted the accused."]
    stub_llm = StubLLM("CLAIM 1: UNSUPPORTED")
    verifier = ClaimVerifier(stub_llm)
    result = verifier.verify_claims(claims, evidence)
    assert result["unsupported_count"] == 1
    assert result["support_rate"] == 0.0
    assert result["verdicts"] == ["UNSUPPORTED"]


def test_claim_verifier_handles_unknown_on_parse_failure():
    claims = ["The fine was Rs. 1,00,000."]
    stub_llm = StubLLM("")  # empty response
    verifier = ClaimVerifier(stub_llm)
    result = verifier.verify_claims(claims, "Some evidence.")
    assert result["verdicts"] == ["UNKNOWN"]
    assert result["unknown_count"] == 1


def test_claim_verifier_empty_claims_vacuous_pass():
    verifier = ClaimVerifier(StubLLM(""))
    result = verifier.verify_claims([], "Some evidence.")
    assert result["support_rate"] == 1.0
    assert result["supported_count"] == 0


# ── VerifiedRAGPipeline abstention tests ─────────────────────────────────

def _make_pipeline(gen_response: str, verify_response: str, chunk_store: dict):
    retriever = StubRetriever([("chunk_001", 0.9), ("chunk_002", 0.8)])
    gen_llm = StubLLM(gen_response)
    verify_llm = StubLLM(verify_response)
    return VerifiedRAGPipeline(
        retriever=retriever,
        chunk_store=chunk_store,
        gen_client=gen_llm,
        verify_client=verify_llm,
        top_k=2,
        abstention_threshold=0.5,
    )


CHUNK_STORE = {
    "chunk_001": {
        "chunk_id": "chunk_001", "case_id": "2015 INSC 1",
        "text": "The court dismissed the appeal.",
        "pages": [1], "paragraphs": ["p1_1"], "sections": ["DECISION"],
        "source_reference": "2015 INSC 1, para p1_1",
    },
    "chunk_002": {
        "chunk_id": "chunk_002", "case_id": "2015 INSC 1",
        "text": "The fine was ordered refunded.",
        "pages": [2], "paragraphs": ["p2_1"], "sections": ["DECISION"],
        "source_reference": "2015 INSC 1, para p2_1",
    },
}


def test_pipeline_answers_when_supported():
    pipeline = _make_pipeline(
        gen_response="The court dismissed the appeal [chunk_001].",
        verify_response="CLAIM 1: SUPPORTED",
        chunk_store=CHUNK_STORE,
    )
    result = pipeline.answer("What happened in the case?")
    assert result["verification"]["abstained"] is False
    assert result["verification"]["support_rate"] == 1.0
    assert "Insufficient evidence" not in result["answer"]


def test_pipeline_abstains_below_threshold():
    pipeline = _make_pipeline(
        gen_response="The court acquitted the accused [chunk_001]. The penalty was waived [chunk_002].",
        verify_response="CLAIM 1: UNSUPPORTED\nCLAIM 2: UNSUPPORTED",
        chunk_store=CHUNK_STORE,
    )
    result = pipeline.answer("What happened?")
    assert result["verification"]["abstained"] is True
    assert result["answer"] == "Insufficient evidence to answer."


def test_pipeline_abstains_on_model_abstention():
    pipeline = _make_pipeline(
        gen_response="Insufficient evidence to answer.",
        verify_response="",
        chunk_store=CHUNK_STORE,
    )
    result = pipeline.answer("What happened?")
    assert result["verification"]["abstained"] is True
    assert result["verification"]["abstention_reason"] == "model_abstained"


def test_pipeline_strips_invalid_citations():
    pipeline = _make_pipeline(
        gen_response="Court ruled [chunk_001] and also see [GHOST_999].",
        verify_response="CLAIM 1: SUPPORTED",
        chunk_store=CHUNK_STORE,
    )
    result = pipeline.answer("What happened?")
    assert "GHOST_999" not in result["answer"]
    assert result["citations"]["citations_stripped"] >= 1


def test_pipeline_records_full_provenance():
    pipeline = _make_pipeline(
        gen_response="The court dismissed the appeal [chunk_001].",
        verify_response="CLAIM 1: SUPPORTED",
        chunk_store=CHUNK_STORE,
    )
    result = pipeline.answer("What happened?")
    assert len(result["retrieved"]) == 2
    assert result["retrieved"][0]["chunk_id"] == "chunk_001"
    assert result["retrieved"][0]["case_id"] == "2015 INSC 1"
    assert result["retrieved"][0]["pages"] == [1]
    assert "verification" in result
    assert "claims" in result["verification"]


# ── Verification metrics aggregation ─────────────────────────────────────

def test_aggregate_includes_verification_metrics():
    """Aggregation should compute mean_support_rate for verified-rag results."""
    results = [
        {
            "evaluation": {"abstention": 0, "correctness": 4, "factuality": 4, "judge_parse_failure": False},
            "generation_failed": False,
            "generation_latency_s": 1.0,
            "citations": {"cited_ids": ["c1"], "invalid_ids": [], "has_citation": True},
            "retrieval_eval": {"gold_hit": True, "gold_recall": 1.0, "cited_gold": True},
            "verification": {
                "support_rate": 1.0, "supported_count": 2, "unsupported_count": 0,
                "unknown_count": 0, "abstained": False, "claims": ["A", "B"],
            },
        },
        {
            "evaluation": {"abstention": 0, "correctness": 3, "factuality": 3, "judge_parse_failure": False},
            "generation_failed": False,
            "generation_latency_s": 1.5,
            "citations": {"cited_ids": ["c2"], "invalid_ids": [], "has_citation": True},
            "retrieval_eval": {"gold_hit": True, "gold_recall": 1.0, "cited_gold": True},
            "verification": {
                "support_rate": 0.5, "supported_count": 1, "unsupported_count": 1,
                "unknown_count": 0, "abstained": False, "claims": ["C", "D"],
            },
        },
    ]
    agg = aggregate(results)
    assert agg["mean_support_rate"] == pytest.approx(0.75)
    assert agg["total_supported_claims"] == 3
    assert agg["total_unsupported_claims"] == 1
    assert agg["verification_abstentions"] == 0

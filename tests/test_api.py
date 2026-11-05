"""
LegalSaarthi — API & Interactive Application Unit Tests (Phase 24)

Tests all FastAPI REST endpoints and service integration:
- Root single-page application delivery
- System health & readiness
- Case search & pagination
- Single-case detail & structural paragraphs
- Chunk provenance lookup
- Question answering (single system & comparison modes)
- Judgment summarization & readability metrics
- Research metrics & benchmark prompts
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.service import LegalSaarthiService


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


def test_serve_root(client):
    """Verify that root URL serves the single-page application HTML."""
    res = client.get("/")
    assert res.status_code == 200
    assert "LegalSaarthi" in res.text
    assert "QA Studio" in res.text


def test_health_check(client):
    """Verify system health endpoint returns 200 and schema."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "total_chunks_loaded" in data


def test_benchmark_prompts(client):
    """Verify curated gold benchmark prompts are returned."""
    res = client.get("/api/benchmark-prompts")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 3
    assert all("prompt" in p and "tier" in p for p in data)


def test_case_search(client):
    """Verify case search with pagination."""
    res = client.get("/api/cases?page=1&page_size=10")
    assert res.status_code == 200
    data = res.json()
    assert data["page"] == 1
    assert data["page_size"] == 10
    assert "total" in data
    assert isinstance(data["cases"], list)
    if data["cases"]:
        c = data["cases"][0]
        assert "case_id" in c
        assert "year" in c


def test_case_detail_found(client):
    """Verify case detail lookup for known case."""
    # 2015 INSC 177 is in the corpus
    res = client.get("/api/cases/2015 INSC 177")
    if res.status_code == 200:
        data = res.json()
        assert data["case_id"] == "2015 INSC 177"
        assert "sections_summary" in data
        assert "total_pages" in data


def test_case_detail_not_found(client):
    """Verify 404 for non-existent case."""
    res = client.get("/api/cases/NON_EXISTENT_CASE_9999")
    assert res.status_code == 404


def test_chunk_provenance_found(client):
    """Verify chunk provenance endpoint with a real or sample chunk."""
    service = LegalSaarthiService.get_instance()
    service.initialize()
    if service.chunks:
        sample_id = next(iter(service.chunks.keys()))
        res = client.get(f"/api/chunks/{sample_id}")
        assert res.status_code == 200
        data = res.json()
        assert data["chunk_id"] == sample_id
        assert "sections" in data
        assert "pages" in data
        assert "paragraphs" in data
        assert "text" in data


def test_chunk_provenance_flexible_resolution(client):
    """Verify flexible user input formats (e.g. '2015 INSC 336 section 9') resolve correctly."""
    # Test human-friendly input without exact underscores
    res = client.get("/api/chunks/2015 INSC 336 section 9")
    if res.status_code == 200:
        data = res.json()
        assert data["chunk_id"] == "2015 INSC 336_section_aware_9"
        assert data["case_id"] == "2015 INSC 336"

    # Test case ID only
    res_case = client.get("/api/chunks/2015 INSC 177")
    if res_case.status_code == 200:
        data = res_case.json()
        assert "2015 INSC 177" in data["chunk_id"]


def test_chunk_provenance_not_found(client):
    """Verify 404 for non-existent chunk ID."""
    res = client.get("/api/chunks/TOTALLY_FAKE_CHUNK_9999")
    assert res.status_code == 404


def test_research_metrics(client):
    """Verify research metrics endpoint."""
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "hypotheses" in data
    assert "baselines" in data
    assert "judge_agreement" in data
    assert data["judge_agreement"]["exact_agreement"] == "50.3%"


def test_qa_endpoint_with_mock(client):
    """Test QA endpoint with mocked LLM output."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "The appeal was dismissed under Section 302 IPC [CHUNK 2015 INSC 177_section_aware_9]."

    service = LegalSaarthiService.get_instance()
    service.initialize()
    original_llm = service.llm_client
    service.llm_client = mock_llm

    try:
        payload = {
            "query": "What was the decision of the Supreme Court?",
            "system": "verified_rag",
            "top_k": 3,
        }
        res = client.post("/api/qa", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["query"] == payload["query"]
        assert "primary" in data
        assert "answer" in data["primary"]
        assert data["primary"]["system"] == "verified_rag"
    finally:
        service.llm_client = original_llm


def test_qa_endpoint_compare_mode(client):
    """Test QA endpoint in compare_all mode."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "The appeal was dismissed [2015 INSC 177_section_aware_9]."

    service = LegalSaarthiService.get_instance()
    service.initialize()
    original_llm = service.llm_client
    service.llm_client = mock_llm

    try:
        payload = {
            "query": "What was the decision?",
            "system": "compare_all",
            "top_k": 3,
        }
        res = client.post("/api/qa", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["comparison"] is not None
        assert "verified_rag" in data["comparison"]
        assert "basic_rag" in data["comparison"]
        assert "generic_llm" in data["comparison"]
    finally:
        service.llm_client = original_llm


def test_summarize_endpoint_with_mock(client):
    """Test summarization endpoint for a valid case."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "FACTS: The appellant was convicted.\nDECISION: The appeal is dismissed."

    service = LegalSaarthiService.get_instance()
    service.initialize()
    original_llm = service.llm_client
    service.llm_client = mock_llm

    try:
        payload = {
            "case_id": "2015 INSC 177",
            "mode": "structured",
        }
        res = client.post("/api/summarize", json=payload)
        if res.status_code == 200:
            data = res.json()
            assert data["case_id"] == "2015 INSC 177"
            assert "primary" in data
            assert data["primary"]["mode"] == "structured"
            assert "flesch_reading_ease" in data["primary"]
            assert data["primary"]["predicted_disposition"] == "dismissed"
    finally:
        service.llm_client = original_llm


# ---------------------------------------------------------------------------
# Phase 24 Audit — Regression Tests for QA Benchmark and Case-Mapping
# ---------------------------------------------------------------------------

def test_regression_benchmark_query_mismatched_case_scope(client):
    """Regression Test 1: Benchmark query mapped to 2015 INSC 942 run against 2015 INSC 1021."""
    mock_llm = MagicMock()
    # Model generates answer using only 1021 chunks or abstains
    mock_llm.generate.return_value = "This matter under 2015 INSC 1021 does not address the Reserve Bank of India [CHUNK 2015 INSC 1021_section_aware_1]."

    service = LegalSaarthiService.get_instance()
    service.initialize()
    original_llm = service.llm_client
    service.llm_client = mock_llm

    try:
        payload = {
            "query": "What statutory powers of the Reserve Bank of India were challenged under the Right to Information Act?",
            "system": "basic_rag",
            "case_id": "2015 INSC 1021",
            "top_k": 5,
        }
        res = client.post("/api/qa", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["retrieval_scope"] == "case_scoped"
        assert data["case_id_filter"] == "2015 INSC 1021"

        # Verify no 2015 INSC 942 chunks were cited
        for c in data["primary"]["cited_chunks"]:
            assert not c["chunk_id"].startswith("2015 INSC 942"), "Cross-case chunk leaked into scoped retrieval!"
    finally:
        service.llm_client = original_llm


def test_regression_case_scoped_retrieval_zero_contamination():
    """Regression Test 2: Case-scoped retrieval strictly contains zero chunks from another case."""
    from src.api.service import FilteredRetrieverWrapper
    service = LegalSaarthiService.get_instance()
    service.initialize()

    target_case = "2015 INSC 177"
    retriever = FilteredRetrieverWrapper(
        service.bm25,
        service.chunks,
        service.case_to_chunks,
        service.chunk_id_to_idx,
        case_id=target_case,
    )

    # Search with keywords intensely associated with a completely different case (2015 INSC 942 RBI case)
    results = retriever.retrieve("Reserve Bank of India Jayantilal N Mistry statutory powers RTI Act", top_k=5)
    assert len(results) > 0, "Expected chunks within target case 2015 INSC 177"
    for cid, score in results:
        chunk = service.chunks.get(cid, {})
        assert chunk.get("case_id") == target_case, f"Contaminated chunk found: {cid} from case {chunk.get('case_id')}"


def test_regression_valid_citation_from_wrong_judgment(client):
    """Regression Test 3: Valid citation ID from the wrong judgment is flagged as case-mismatch."""
    mock_llm = MagicMock()
    # Model cites a real, valid chunk in the database (2015 INSC 942_section_aware_14),
    # but the request was explicitly scoped to 2015 INSC 177!
    mock_llm.generate.return_value = "The bank argued statutory privilege [CHUNK 2015 INSC 942_section_aware_14]."

    service = LegalSaarthiService.get_instance()
    service.initialize()
    original_llm = service.llm_client
    service.llm_client = mock_llm

    try:
        payload = {
            "query": "What did the bank argue?",
            "system": "basic_rag",
            "case_id": "2015 INSC 177",
            "top_k": 5,
        }
        res = client.post("/api/qa", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["primary"]["case_fidelity_passed"] is False, "Cross-case citation must trigger case_fidelity failure"
        assert "2015 INSC 942_section_aware_14" in data["primary"]["mismatched_citation_ids"]

        # Check citation item flags
        mismatched_items = [c for c in data["primary"]["cited_chunks"] if not c["case_matches_scope"]]
        assert len(mismatched_items) == 1
        assert mismatched_items[0]["case_id"] == "2015 INSC 942"
        assert mismatched_items[0]["case_matches_scope"] is False
    finally:
        service.llm_client = original_llm


def test_regression_global_search_legitimately_retrieves_other_case(client):
    """Regression Test 4: Global search query legitimately retrieves cases without mismatch warning."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "The Reserve Bank of India inspection powers were tested [CHUNK 2015 INSC 942_section_aware_14]."

    service = LegalSaarthiService.get_instance()
    service.initialize()
    original_llm = service.llm_client
    service.llm_client = mock_llm

    try:
        payload = {
            "query": "What statutory powers of the Reserve Bank of India were challenged under the Right to Information Act?",
            "system": "basic_rag",
            "case_id": None,  # Explicit global retrieval
            "top_k": 5,
        }
        res = client.post("/api/qa", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["retrieval_scope"] == "global"
        assert data["case_id_filter"] is None
        assert data["primary"]["case_fidelity_passed"] is True
        assert data["primary"]["mismatched_citation_ids"] == []
    finally:
        service.llm_client = original_llm


def test_regression_missing_or_invalid_gold_case_mapping(client):
    """Regression Test 5: Missing or invalid case mapping returns 0 chunks and safely abstains without leaking global corpus."""
    from src.api.service import FilteredRetrieverWrapper
    service = LegalSaarthiService.get_instance()
    service.initialize()

    invalid_case = "INVALID_CASE_NON_EXISTENT_9999"
    retriever = FilteredRetrieverWrapper(
        service.bm25,
        service.chunks,
        service.case_to_chunks,
        service.chunk_id_to_idx,
        case_id=invalid_case,
    )

    # Must return empty list, NEVER fall back to global
    res_chunks = retriever.retrieve("Supreme Court order disposition", top_k=5)
    assert res_chunks == [], "Invalid case must yield empty retrieval list, not global fallback"

    # API call with invalid case ID with mocked LLM
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "No information is available in this document."
    original_llm = service.llm_client
    service.llm_client = mock_llm

    try:
        payload = {
            "query": "What was the final disposition?",
            "system": "basic_rag",
            "case_id": invalid_case,
            "top_k": 5,
        }
        res = client.post("/api/qa", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["retrieval_scope"] == "case_scoped"
        assert data["case_id_filter"] == invalid_case
        assert data["primary"]["case_fidelity_passed"] is True
    finally:
        service.llm_client = original_llm

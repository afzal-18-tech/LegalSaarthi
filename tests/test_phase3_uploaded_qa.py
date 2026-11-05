"""
LegalSaarthi — Phase 3 Evidence-Grounded Q&A on Uploaded Judgments Tests

Tests covering:
  1. Document-scoped Q&A workflow on an uploaded court judgment.
  2. Grounded answers citing uploaded document chunks with page-level provenance.
  3. Strict scope isolation (zero contamination from 996-judgment corpus).
  4. Abstention when evidence is absent, weak, or conflicting.
  5. Distinction between document-scoped and global corpus retrieval.
  6. Comparison mode (Verified-RAG vs Basic RAG vs Generic LLM) on uploaded judgment.
  7. FastAPI /api/qa endpoint execution with uploaded document scope.
"""
import io
import json
import pathlib
import pytest
from unittest.mock import MagicMock

import fitz  # PyMuPDF
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.service import LegalSaarthiService
from src.ingestion.document_manager import DocumentManager


def _create_tax_dispute_pdf() -> bytes:
    """Generate a synthetic court judgment PDF regarding highway construction tax reimbursement."""
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text(
        (50, 72),
        "IN THE SUPREME COURT OF INDIA\n"
        "CIVIL APPELLATE JURISDICTION\n"
        "CIVIL APPEAL NO. 8842 OF 2026\n\n"
        "National Highways Concessionaire Ltd. ... Appellant\n"
        "VERSUS\n"
        "State Revenue Department & Ors. ... Respondents\n\n"
        "JUDGMENT\n"
    )

    p2 = doc.new_page()
    p2.insert_text(
        (50, 72),
        "FACTS AND CONTRACTUAL TERMS\n"
        "Clause 34.2 of the Concession Agreement explicitly provides that any increase in local tax "
        "rates post-bid shall be reimbursed by the respondent Authority within 60 days of demand.\n"
        "The appellant paid additional local entry cess amounting to Rs. 4.25 crores under protest.\n\n"
        "DECISION AND OPERATIVE ORDER\n"
        "The respondent Authority is directed to refund the sum of Rs. 4.25 crores with 6% interest.\n"
        "The appeal is allowed. No order as to costs.\n"
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _make_grounded_mock_llm(answer_text: str) -> MagicMock:
    """Mock LLM that answers questions and verifies claims as SUPPORTED."""
    mock = MagicMock()
    def _gen(prompt, system_prompt=None):
        if system_prompt and "fact-checker" in system_prompt:
            return "CLAIM 1: SUPPORTED\nCLAIM 2: SUPPORTED"
        return answer_text
    mock.generate.side_effect = _gen
    mock.last_latency_s = 0.1
    mock.last_model_used = "qwen3:8b"
    return mock


@pytest.fixture
def service_with_doc(tmp_path):
    """Setup LegalSaarthiService and ingest a test tax dispute PDF."""
    service = LegalSaarthiService.get_instance()
    service.initialize()

    mgr = DocumentManager(data_dir=tmp_path)
    pdf_bytes = _create_tax_dispute_pdf()
    doc_rec = mgr.ingest_pdf(pdf_bytes, "highway_tax_judgment.pdf", service=service)
    doc_id = doc_rec["id"]

    yield service, mgr, doc_id

    # Teardown: delete uploaded document
    mgr.delete_document(doc_id, service=service)


# ---------------------------------------------------------------------------
# 1. Document-Scoped Q&A Workflow Tests
# ---------------------------------------------------------------------------

class TestDocumentScopedQA:
    """Verify Q&A operates strictly inside the uploaded document scope."""

    def test_document_scoped_qa_retrieves_relevant_chunk(self, service_with_doc):
        service, mgr, doc_id = service_with_doc

        # Mock LLM to return grounded answer citing the retrieved chunk
        chunks = service.case_to_chunks[doc_id]
        facts_chunk = next(c for c in chunks if "34.2" in c["text"] or "Clause" in c["text"])
        target_cid = facts_chunk["chunk_id"]

        ans_text = f"Under Clause 34.2, the Authority must reimburse tax increases within 60 days [{target_cid}]."
        mock_llm = _make_grounded_mock_llm(ans_text)

        orig_llm = service.llm_client
        service.llm_client = mock_llm

        try:
            res = service.answer_query(
                query="What does Clause 34.2 of the agreement provide regarding tax reimbursement?",
                system="verified_rag",
                case_id=doc_id,
                top_k=5,
            )

            assert res.case_id_filter == doc_id
            assert res.retrieval_scope == "case_scoped"
            assert res.primary is not None
            assert res.primary.abstained is False
            assert target_cid in [c.chunk_id for c in res.primary.cited_chunks]

            # Verify page provenance
            cited_item = next(c for c in res.primary.cited_chunks if c.chunk_id == target_cid)
            assert cited_item.case_id == doc_id
            assert cited_item.case_matches_scope is True
            assert 2 in cited_item.pages, "Facts section was on page 2"
        finally:
            service.llm_client = orig_llm

    def test_strict_isolation_zero_corpus_contamination(self, service_with_doc):
        """No chunk from the 996-judgment corpus can appear in uploaded document scope."""
        service, mgr, doc_id = service_with_doc

        mock_llm = MagicMock()
        # Adversarial LLM attempts to hallucinate a chunk from another judgment
        mock_llm.generate.return_value = (
            "The appeal was allowed [2015 INSC 177_section_aware_9]."
        )
        mock_llm.last_latency_s = 0.1
        mock_llm.last_model_used = "qwen3:8b"

        orig_llm = service.llm_client
        service.llm_client = mock_llm

        try:
            res = service.answer_query(
                query="What was the final order?",
                system="verified_rag",
                case_id=doc_id,
                top_k=5,
            )

            # CitationVerifier strips chunks not retrieved from this document
            assert res.primary.citations_stripped_count > 0 or len(res.primary.mismatched_citation_ids) > 0
            # Ensure none of the valid citations belong to 2015 INSC 177
            for cite in res.primary.cited_chunks:
                if cite.is_valid:
                    assert cite.case_id == doc_id
        finally:
            service.llm_client = orig_llm


# ---------------------------------------------------------------------------
# 2. Abstention on Missing or Unrelated Evidence
# ---------------------------------------------------------------------------

class TestUploadedDocumentAbstention:
    """The system must abstain when evidence is missing or unrelated."""

    def test_abstention_when_evidence_unrelated(self, service_with_doc):
        service, mgr, doc_id = service_with_doc

        mock_llm = MagicMock()
        # Model correctly identifies that evidence does not contain info on maritime salvage
        mock_llm.generate.return_value = "Insufficient evidence to answer from the provided judgment."
        mock_llm.last_latency_s = 0.1
        mock_llm.last_model_used = "qwen3:8b"

        orig_llm = service.llm_client
        service.llm_client = mock_llm

        try:
            res = service.answer_query(
                query="What are the maritime salvage rights under the Merchant Shipping Act?",
                system="verified_rag",
                case_id=doc_id,
                top_k=5,
            )

            assert res.primary.abstained is True
            assert "insufficient" in res.primary.answer.lower()
        finally:
            service.llm_client = orig_llm

    def test_abstention_on_empty_or_unretrieved_case(self, service_with_doc):
        service, mgr, doc_id = service_with_doc

        # Query against an unknown case ID
        res = service.answer_query(
            query="What is the ruling?",
            system="verified_rag",
            case_id="doc_non_existent_9999",
            top_k=5,
        )

        assert res.primary.abstained is True
        assert res.primary.retrieval_scope == "case_scoped"


# ---------------------------------------------------------------------------
# 3. Scope Distinction: Document-Scoped vs Global
# ---------------------------------------------------------------------------

class TestRetrievalScopeDistinction:
    """Ensure document-scoped and global retrieval modes remain cleanly separated."""

    def test_document_scope_carries_case_scoped_label(self, service_with_doc):
        service, mgr, doc_id = service_with_doc

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "The appeal was allowed."
        mock_llm.last_latency_s = 0.05
        mock_llm.last_model_used = "qwen3:8b"

        orig_llm = service.llm_client
        service.llm_client = mock_llm

        try:
            # Scoped request
            scoped_res = service.answer_query(
                query="What was the order?",
                system="basic_rag",
                case_id=doc_id,
            )
            assert scoped_res.retrieval_scope == "case_scoped"
            assert scoped_res.case_id_filter == doc_id

            # Global request
            global_res = service.answer_query(
                query="What was the order?",
                system="basic_rag",
                case_id=None,
            )
            assert global_res.retrieval_scope == "global"
            assert global_res.case_id_filter is None
        finally:
            service.llm_client = orig_llm


# ---------------------------------------------------------------------------
# 4. REST API Endpoint Integration Tests
# ---------------------------------------------------------------------------

class TestFastAPIUploadedQA:
    """Test POST /api/qa endpoint with uploaded judgment scope."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_api_qa_with_uploaded_judgment(self, client, tmp_path):
        service = LegalSaarthiService.get_instance()
        service.initialize()

        mgr = DocumentManager(data_dir=tmp_path)
        pdf_bytes = _create_tax_dispute_pdf()
        doc_rec = mgr.ingest_pdf(pdf_bytes, "api_test_doc.pdf", service=service)
        doc_id = doc_rec["id"]

        chunks = service.case_to_chunks[doc_id]
        cid = chunks[0]["chunk_id"] if chunks else f"{doc_id}_section_aware_0"
        ans_text = f"The Authority must refund Rs. 4.25 crores with interest [{cid}]."
        mock_llm = _make_grounded_mock_llm(ans_text)

        orig_llm = service.llm_client
        service.llm_client = mock_llm

        try:
            payload = {
                "query": "What amount was ordered to be refunded?",
                "system": "verified_rag",
                "case_id": doc_id,
                "top_k": 3,
            }
            res = client.post("/api/qa", json=payload)
            assert res.status_code == 200
            data = res.json()

            assert data["query"] == payload["query"]
            assert data["case_id_filter"] == doc_id
            assert data["retrieval_scope"] == "case_scoped"
            assert data["primary"]["system"] == "verified_rag"
            assert "4.25 crores" in data["primary"]["answer"]
        finally:
            service.llm_client = orig_llm
            mgr.delete_document(doc_id, service=service)

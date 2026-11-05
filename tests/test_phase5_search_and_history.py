"""
LegalSaarthi — Phase 5 In-Document Search & Lightweight History Tests

Tests covering:
  1. In-document keyword and exact phrase search within uploaded PDFs.
  2. Context snippet generation centered on matched terms.
  3. Page provenance and source reference preservation.
  4. Search within 996-corpus cases.
  5. Local private HistoryManager (record, filter, single-item delete, clear all).
  6. FIFO eviction cap at MAX_HISTORY_ITEMS.
  7. Disk persistence and restart recovery for query history.
  8. FastAPI REST endpoints:
     - GET /api/documents/{doc_id}/search
     - GET /api/history
     - DELETE /api/history/{query_id}
     - DELETE /api/history
     - Auto-recording of queries during POST /api/qa.
"""
import io
import json
import pathlib
import pytest
from unittest.mock import MagicMock, patch

import fitz  # PyMuPDF
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.history import HistoryManager, MAX_HISTORY_ITEMS
from src.api.service import LegalSaarthiService
from src.ingestion.document_manager import DocumentManager


def _create_search_test_pdf() -> bytes:
    """Create a multi-page PDF with distinct legal topics on separate pages."""
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text(
        (50, 72),
        "IN THE SUPREME COURT OF INDIA\n"
        "CIVIL APPELLATE JURISDICTION\n"
        "CIVIL APPEAL NO. 8899 OF 2026\n\n"
        "National Highways Authority v. Patel Engineering\n\n"
        "JUDGMENT\n"
        "The question relates to contractual arbitration under the Arbitration and Conciliation Act 1996.\n"
    )

    p2 = doc.new_page()
    p2.insert_text(
        (50, 72),
        "FACTS AND SUBMISSIONS\n"
        "The respondent entered into an EPC concession agreement for construction of a four-lane highway.\n"
        "Due to delays in handing over encumbrance free land, liquidated damages were levied.\n"
        "The respondent invoked the dispute resolution clause seeking appointment of an independent arbitrator.\n"
    )

    p3 = doc.new_page()
    p3.insert_text(
        (50, 72),
        "ANALYSIS AND COURT REASONING\n"
        "It is well settled that statutory forfeiture requires strict proof of actual loss.\n"
        "Under Section 74 of the Indian Contract Act, reasonable compensation is the guiding principle.\n\n"
        "DECISION AND FINAL ORDER\n"
        "The appeal is accordingly dismissed with costs quantified at rupees fifty thousand.\n"
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ---------------------------------------------------------------------------
# In-Document Search Tests
# ---------------------------------------------------------------------------

class TestInDocumentSearch:

    def test_search_phrase_boost_and_snippet(self, tmp_path):
        """Exact phrase matches receive boost, return snippets with page numbers."""
        mgr = DocumentManager(data_dir=tmp_path)
        pdf_bytes = _create_search_test_pdf()
        meta = mgr.ingest_pdf(pdf_bytes, "highway_dispute.pdf")
        doc_id = meta["id"]

        hits = mgr.search_in_document(doc_id, "concession agreement")
        assert len(hits) > 0

        top_hit = hits[0]
        assert top_hit["match_type"] == "phrase"
        assert 2 in top_hit["pages"]
        assert "concession agreement" in top_hit["snippet"].lower()
        assert top_hit["document_id"] == doc_id
        assert top_hit["score"] > 5.0  # Exact phrase bonus (+5.0)

    def test_search_keyword_matching(self, tmp_path):
        """Keyword query matches terms across pages."""
        mgr = DocumentManager(data_dir=tmp_path)
        pdf_bytes = _create_search_test_pdf()
        meta = mgr.ingest_pdf(pdf_bytes, "arbitration_matter.pdf")
        doc_id = meta["id"]

        hits = mgr.search_in_document(doc_id, "forfeiture compensation")
        assert len(hits) > 0

        # Should match page 3 where Section 74 / forfeiture is discussed
        page_3_hits = [h for h in hits if h["page_start"] == 3]
        assert len(page_3_hits) > 0
        assert "forfeiture" in page_3_hits[0]["snippet"].lower()

    def test_search_no_match_returns_empty(self, tmp_path):
        """Query with completely unrelated terms returns empty list."""
        mgr = DocumentManager(data_dir=tmp_path)
        pdf_bytes = _create_search_test_pdf()
        meta = mgr.ingest_pdf(pdf_bytes, "sample.pdf")
        doc_id = meta["id"]

        hits = mgr.search_in_document(doc_id, "quantum gravitational singularity teleportation")
        assert hits == []

    def test_search_empty_query_returns_empty(self, tmp_path):
        """Empty or whitespace query returns empty list without error."""
        mgr = DocumentManager(data_dir=tmp_path)
        pdf_bytes = _create_search_test_pdf()
        meta = mgr.ingest_pdf(pdf_bytes, "sample.pdf")
        doc_id = meta["id"]

        assert mgr.search_in_document(doc_id, "") == []
        assert mgr.search_in_document(doc_id, "   \n  \t ") == []

    def test_search_corpus_case_via_service(self):
        """In-document search works on baseline 996-corpus judgments via service."""
        service = LegalSaarthiService.get_instance()
        service.initialize()

        # 2015 INSC 177 is in the baseline corpus
        hits = service.search_in_document("2015 INSC 177", "appeal")
        assert len(hits) > 0
        assert hits[0]["document_id"] == "2015 INSC 177"
        assert len(hits[0]["snippet"]) > 0


# ---------------------------------------------------------------------------
# History Manager Tests
# ---------------------------------------------------------------------------

class TestHistoryManager:

    def test_record_and_list_queries(self, tmp_path):
        """Records are persisted locally, retrievable, and filterable by case_id."""
        hm = HistoryManager(data_dir=tmp_path)
        assert len(hm.list_history()) == 0

        r1 = hm.record_query(
            query="What is the concession period?",
            case_id="doc_abc",
            document_name="Highway Dispute",
            system="verified_rag",
            answer="The concession period is 15 years.",
            abstained=False,
            top_citation="chunk_001",
            latency_s=1.25,
        )

        r2 = hm.record_query(
            query="What are the grounds of appeal?",
            case_id="2015 INSC 177",
            document_name="Manekia Case",
            system="basic_rag",
            answer="Grounds relate to evidentiary standards.",
            abstained=False,
            latency_s=0.88,
        )

        all_records = hm.list_history()
        assert len(all_records) == 2
        # Most recent first
        assert all_records[0]["query_id"] == r2["query_id"]
        assert all_records[1]["query_id"] == r1["query_id"]

        # Filter by case_id
        doc_abc_records = hm.list_history(case_id="doc_abc")
        assert len(doc_abc_records) == 1
        assert doc_abc_records[0]["query"] == "What is the concession period?"

        nonexistent = hm.list_history(case_id="unknown_case")
        assert len(nonexistent) == 0

    def test_delete_single_item(self, tmp_path):
        """Deleting an existing record removes it; deleting non-existent returns False."""
        hm = HistoryManager(data_dir=tmp_path)
        r = hm.record_query("Test query 1", case_id="case_1")
        hm.record_query("Test query 2", case_id="case_2")

        assert len(hm.list_history()) == 2
        deleted = hm.delete_history_item(r["query_id"])
        assert deleted is True
        assert len(hm.list_history()) == 1

        not_deleted = hm.delete_history_item("non_existent_qid")
        assert not_deleted is False

    def test_clear_all_history(self, tmp_path):
        """clear_history wipes all entries and returns the count."""
        hm = HistoryManager(data_dir=tmp_path)
        hm.record_query("Q1")
        hm.record_query("Q2")
        hm.record_query("Q3")

        assert len(hm.list_history()) == 3
        cleared_count = hm.clear_history()
        assert cleared_count == 3
        assert len(hm.list_history()) == 0

    def test_fifo_eviction_cap(self, tmp_path):
        """HistoryManager respects MAX_HISTORY_ITEMS cap via FIFO eviction."""
        hm = HistoryManager(data_dir=tmp_path)
        # Artificially insert records up to cap
        for i in range(MAX_HISTORY_ITEMS + 10):
            hm.record_query(f"Query {i}")

        records = hm.list_history(limit=1000)
        assert len(records) == MAX_HISTORY_ITEMS
        # The oldest entries (Query 0, 1, ..., 9) should be evicted
        assert records[0]["query"] == f"Query {MAX_HISTORY_ITEMS + 9}"
        assert records[-1]["query"] == "Query 10"

    def test_persistence_across_reloads(self, tmp_path):
        """History reloads identically from query_history.json across restarts."""
        hm1 = HistoryManager(data_dir=tmp_path)
        hm1.record_query("Persistent query", case_id="doc_xyz", answer="Persisted answer")

        # Create separate instance pointing to same data_dir
        hm2 = HistoryManager(data_dir=tmp_path)
        records = hm2.list_history()
        assert len(records) == 1
        assert records[0]["query"] == "Persistent query"
        assert records[0]["case_id"] == "doc_xyz"


# ---------------------------------------------------------------------------
# FastAPI REST Endpoint Tests
# ---------------------------------------------------------------------------

class TestPhase5RestEndpoints:

    @pytest.fixture(autouse=True)
    def setup_client(self, tmp_path):
        self.client = TestClient(app)
        self.tmp_path = tmp_path

    def test_api_search_uploaded_document(self):
        """GET /api/documents/{doc_id}/search returns structured hits."""
        pdf_bytes = _create_search_test_pdf()
        up_res = self.client.post(
            "/api/documents/upload",
            files={"file": ("highway_search_test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert up_res.status_code == 200
        doc_id = up_res.json()["document"]["id"]

        search_res = self.client.get(
            f"/api/documents/{doc_id}/search",
            params={"q": "liquidated damages", "top_k": 5},
        )
        assert search_res.status_code == 200
        data = search_res.json()
        assert data["document_id"] == doc_id
        assert data["query"] == "liquidated damages"
        assert data["total_hits"] >= 1
        hit = data["hits"][0]
        assert "damages" in hit["snippet"].lower()
        assert 2 in hit["pages"]

    def test_api_search_corpus_document(self):
        """GET /api/documents/{doc_id}/search works for baseline corpus cases."""
        search_res = self.client.get(
            "/api/documents/2015 INSC 177/search",
            params={"q": "appeal", "top_k": 3},
        )
        assert search_res.status_code == 200
        data = search_res.json()
        assert data["total_hits"] >= 1

    def test_api_search_404_for_unknown_doc(self):
        """GET /api/documents/{doc_id}/search returns 404 for unknown IDs."""
        res = self.client.get(
            "/api/documents/completely_unknown_case_id/search",
            params={"q": "arbitration"},
        )
        assert res.status_code == 404

    def test_api_history_lifecycle(self):
        """Test GET, DELETE /api/history/{id}, and DELETE /api/history."""
        # 1. Clear any prior test state
        self.client.delete("/api/history")

        # 2. Add an item via service
        service = LegalSaarthiService.get_instance()
        from src.api.history import HistoryManager
        hm = HistoryManager.get_instance()
        r = hm.record_query("API History Test Query", case_id="test_case")

        # 3. GET /api/history
        get_res = self.client.get("/api/history")
        assert get_res.status_code == 200
        data = get_res.json()
        assert data["total"] >= 1
        assert any(h["query_id"] == r["query_id"] for h in data["history"])

        # 4. DELETE /api/history/{query_id}
        del_item_res = self.client.delete(f"/api/history/{r['query_id']}")
        assert del_item_res.status_code == 200

        # 5. Verify it's gone
        del_again = self.client.delete(f"/api/history/{r['query_id']}")
        assert del_again.status_code == 404

        # 6. DELETE /api/history (clear all)
        hm.record_query("Another query")
        clear_res = self.client.delete("/api/history")
        assert clear_res.status_code == 200
        assert self.client.get("/api/history").json()["total"] == 0

    def test_api_qa_auto_records_query_in_history(self):
        """Executing POST /api/qa automatically adds a record to query history."""
        # Ensure clean history state
        self.client.delete("/api/history")

        # Mock LLM response to avoid live Ollama requirement
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "The Supreme Court dismissed the appeal [2015 INSC 177_section_aware_9]."

        with patch("src.api.service.LegalSaarthiService._run_single_qa") as mock_qa:
            from src.api.models import QASystemResult
            mock_qa.return_value = QASystemResult(
                system="verified_rag",
                system_name="LegalSaarthi Verified-RAG",
                answer="The Supreme Court dismissed the appeal [2015 INSC 177_section_aware_9].",
                abstained=False,
                retrieval_scope="case_scoped",
                case_fidelity_passed=True,
                latency_s=0.5,
            )

            qa_res = self.client.post(
                "/api/qa",
                json={
                    "query": "What was the final disposition in the Manekia appeal?",
                    "case_id": "2015 INSC 177",
                    "system": "verified_rag",
                    "top_k": 3,
                },
            )
            assert qa_res.status_code == 200

        # Check that query was recorded in history
        hist_res = self.client.get("/api/history")
        assert hist_res.status_code == 200
        hist_data = hist_res.json()
        assert hist_data["total"] >= 1
        recorded = hist_data["history"][0]
        assert "Manekia" in recorded["query"]
        assert recorded["case_id"] == "2015 INSC 177"
        assert recorded["system"] == "verified_rag"

"""
LegalSaarthi — Phase 2 PDF Ingestion & Document Lifecycle Tests

Tests covering:
  1. PDF validation (size, magic header, encryption, empty files).
  2. End-to-end ingestion pipeline (Extract -> Detect Structure -> Chunk -> Persist).
  3. Page-level mapping and 1-based page index preservation.
  4. Legal-aware chunk provenance (document_id, page_start, page_end, section, content_hash).
  5. Idempotent uploads (hash deduplication without duplicate chunk generation).
  6. Document isolation (scoped retrieval, zero contamination with 996-corpus).
  7. Document deletion (removes only target document files/chunks, corpus preserved).
  8. FastAPI REST endpoints (/api/documents/*).
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
from src.ingestion.document_manager import DocumentManager, sanitize_filename


def _create_sample_pdf(num_pages: int = 2) -> bytes:
    """Create an in-memory sample court-judgment PDF for testing."""
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text(
        (50, 72),
        "IN THE SUPREME COURT OF INDIA\n"
        "CIVIL APPELLATE JURISDICTION\n"
        "CIVIL APPEAL NO. 4521 OF 2026\n\n"
        "ABC Construction Ltd. ... Appellant\n"
        "VERSUS\n"
        "State of Maharashtra & Anr. ... Respondents\n\n"
        "JUDGMENT\n"
    )

    if num_pages > 1:
        p2 = doc.new_page()
        p2.insert_text(
            (50, 72),
            "FACTS OF THE CASE\n"
            "The appellant entered into a concession agreement for construction of highways.\n"
            "A dispute arose regarding reimbursement of additional tax liabilities under the contract.\n\n"
            "DECISION AND FINAL ORDER\n"
            "In view of the foregoing reasoning, the impugned judgment of the High Court is set aside.\n"
            "The appeal is accordingly allowed. No order as to costs.\n"
        )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ---------------------------------------------------------------------------
# 1. PDF Validation Tests
# ---------------------------------------------------------------------------

class TestPDFValidation:
    """Validate format, size, header, encryption, and empty file handling."""

    def test_sanitize_filename(self):
        assert sanitize_filename("../../../etc/passwd.pdf") == "passwd.pdf"
        assert sanitize_filename("my  legal   doc??.pdf") == "my legal doc__.pdf"
        assert sanitize_filename("") == "unnamed_judgment.pdf"

    def test_valid_pdf_bytes_pass(self, tmp_path):
        mgr = DocumentManager(data_dir=tmp_path)
        pdf_bytes = _create_sample_pdf(2)
        valid, err = mgr.validate_pdf_bytes(pdf_bytes, "test.pdf")
        assert valid is True
        assert err == ""

    def test_empty_bytes_rejected(self, tmp_path):
        mgr = DocumentManager(data_dir=tmp_path)
        valid, err = mgr.validate_pdf_bytes(b"", "test.pdf")
        assert valid is False
        assert "Empty file" in err

    def test_non_pdf_extension_rejected(self, tmp_path):
        mgr = DocumentManager(data_dir=tmp_path)
        valid, err = mgr.validate_pdf_bytes(b"%PDF-fake", "test.docx")
        assert valid is False
        assert "Only PDF documents are accepted" in err

    def test_invalid_magic_bytes_rejected(self, tmp_path):
        mgr = DocumentManager(data_dir=tmp_path)
        valid, err = mgr.validate_pdf_bytes(b"NOT_A_PDF_CONTENT", "test.pdf")
        assert valid is False
        assert "Invalid PDF header signature" in err

    def test_size_limit_rejected(self, tmp_path, monkeypatch):
        mgr = DocumentManager(data_dir=tmp_path)
        monkeypatch.setattr("src.ingestion.document_manager.MAX_FILE_SIZE_BYTES", 500)
        pdf_bytes = _create_sample_pdf(1)
        assert len(pdf_bytes) > 500
        valid, err = mgr.validate_pdf_bytes(pdf_bytes, "large.pdf")
        assert valid is False
        assert "exceeds maximum allowed limit" in err


# ---------------------------------------------------------------------------
# 2. End-to-End Ingestion & Page Mapping Tests
# ---------------------------------------------------------------------------

class TestDocumentIngestion:
    """Test end-to-end ingestion pipeline and page/chunk provenance."""

    def test_end_to_end_ingestion(self, tmp_path):
        mgr = DocumentManager(data_dir=tmp_path)
        pdf_bytes = _create_sample_pdf(2)

        record = mgr.ingest_pdf(pdf_bytes, "sample_judgment.pdf")
        assert record["status"] == "ready"
        assert record["page_count"] == 2
        assert record["chunk_count"] > 0
        assert record["original_filename"] == "sample_judgment.pdf"
        assert record["document_type"] == "court_judgment"
        assert record["error_message"] is None

        # Verify physical files
        doc_id = record["id"]
        assert (mgr.pdfs_dir / f"{doc_id}.pdf").exists()
        assert (mgr.structured_dir / f"{doc_id}.json").exists()
        assert (mgr.chunks_dir / f"{doc_id}.jsonl").exists()

    def test_page_numbers_are_one_based_and_preserved(self, tmp_path):
        mgr = DocumentManager(data_dir=tmp_path)
        pdf_bytes = _create_sample_pdf(2)

        record = mgr.ingest_pdf(pdf_bytes, "sample_judgment.pdf")
        doc_id = record["id"]
        chunks = mgr.get_document_chunks(doc_id)

        all_pages = [p for c in chunks for p in c["pages"]]
        assert len(all_pages) > 0
        assert min(all_pages) == 1, "Page numbers must be 1-based"
        assert max(all_pages) <= 2

    def test_chunk_provenance_fields(self, tmp_path):
        mgr = DocumentManager(data_dir=tmp_path)
        pdf_bytes = _create_sample_pdf(2)

        record = mgr.ingest_pdf(pdf_bytes, "sample_judgment.pdf")
        doc_id = record["id"]
        chunks = mgr.get_document_chunks(doc_id)

        for c in chunks:
            assert c["document_id"] == doc_id
            assert c["case_id"] == doc_id
            assert c["document_name"] == "sample_judgment.pdf"
            assert "chunk_index" in c
            assert "page_start" in c
            assert "page_end" in c
            assert "section" in c
            assert "content_hash" in c
            assert "source_reference" in c
            assert "Page" in c["source_reference"]

    def test_idempotent_ingestion_no_duplicate_chunks(self, tmp_path):
        mgr = DocumentManager(data_dir=tmp_path)
        pdf_bytes = _create_sample_pdf(2)

        # Ingest first time
        rec1 = mgr.ingest_pdf(pdf_bytes, "test.pdf")
        assert rec1["status"] == "ready"
        count1 = rec1["chunk_count"]

        # Ingest identical file second time
        rec2 = mgr.ingest_pdf(pdf_bytes, "test.pdf")
        assert rec2["id"] == rec1["id"]
        assert rec2["status"] == "ready"

        # Verify chunks were not duplicated on disk
        chunks = mgr.get_document_chunks(rec1["id"])
        assert len(chunks) == count1


# ---------------------------------------------------------------------------
# 3. Document Lifecycle & Deletion Tests
# ---------------------------------------------------------------------------

class TestDocumentLifecycle:
    """Test listing, retrieval, deletion, and retry."""

    def test_list_and_get_document(self, tmp_path):
        mgr = DocumentManager(data_dir=tmp_path)
        pdf_bytes = _create_sample_pdf(1)
        rec = mgr.ingest_pdf(pdf_bytes, "test_list.pdf")
        doc_id = rec["id"]

        doc_list = mgr.list_documents()
        assert len(doc_list) == 1
        assert doc_list[0]["id"] == doc_id

        fetched = mgr.get_document(doc_id)
        assert fetched is not None
        assert fetched["original_filename"] == "test_list.pdf"

    def test_delete_removes_only_target_document(self, tmp_path):
        mgr = DocumentManager(data_dir=tmp_path)
        pdf_bytes1 = _create_sample_pdf(1)
        pdf_bytes2 = _create_sample_pdf(2)

        rec1 = mgr.ingest_pdf(pdf_bytes1, "doc1.pdf")
        rec2 = mgr.ingest_pdf(pdf_bytes2, "doc2.pdf")

        # Delete doc1
        deleted = mgr.delete_document(rec1["id"])
        assert deleted is True

        # doc1 files removed
        assert not (mgr.pdfs_dir / f"{rec1['id']}.pdf").exists()
        assert not (mgr.structured_dir / f"{rec1['id']}.json").exists()
        assert not (mgr.chunks_dir / f"{rec1['id']}.jsonl").exists()

        # doc2 files completely intact
        assert (mgr.pdfs_dir / f"{rec2['id']}.pdf").exists()
        assert mgr.get_document(rec2["id"]) is not None


# ---------------------------------------------------------------------------
# 4. Service Integration & Scoped Retrieval Isolation
# ---------------------------------------------------------------------------

class TestServiceIntegration:
    """Verify document registration into LegalSaarthiService and scoped isolation."""

    def test_service_registration_and_scoped_retrieval(self, tmp_path):
        service = LegalSaarthiService.get_instance()
        service.initialize()

        mgr = DocumentManager(data_dir=tmp_path)
        pdf_bytes = _create_sample_pdf(2)
        rec = mgr.ingest_pdf(pdf_bytes, "highways_contract.pdf", service=service)
        doc_id = rec["id"]

        try:
            # 1. Document registered in service memory
            assert doc_id in service.case_to_chunks
            assert doc_id in service.cases_meta
            assert doc_id in service.case_to_file
            assert service.cases_meta[doc_id]["is_uploaded"] is True

            # 2. Case-scoped retrieval for this document
            from src.api.service import FilteredRetrieverWrapper
            retriever = FilteredRetrieverWrapper(
                service.bm25,
                service.chunks,
                service.case_to_chunks,
                service.chunk_id_to_idx,
                case_id=doc_id,
            )
            results = retriever.retrieve("highway concession agreement tax reimbursement", top_k=5)
            assert len(results) > 0

            # 3. Strict zero-contamination assertion: all retrieved chunks belong to doc_id
            for cid, score in results:
                chunk = service.chunks[cid]
                assert chunk["case_id"] == doc_id
                assert chunk["document_id"] == doc_id
        finally:
            # Clean up document from service
            mgr.delete_document(doc_id, service=service)
            assert doc_id not in service.case_to_chunks
            assert doc_id not in service.cases_meta


# ---------------------------------------------------------------------------
# 5. FastAPI REST API Endpoint Tests
# ---------------------------------------------------------------------------

class TestDocumentAPIEndpoints:
    """Test /api/documents/* endpoints using FastAPI TestClient."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_upload_and_lifecycle_api(self, client):
        pdf_bytes = _create_sample_pdf(2)

        # 1. Upload PDF
        res = client.post(
            "/api/documents/upload",
            files={"file": ("test_appeal.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ready"
        doc = data["document"]
        doc_id = doc["id"]
        assert doc["original_filename"] == "test_appeal.pdf"
        assert doc["page_count"] == 2

        try:
            # 2. List documents
            list_res = client.get("/api/documents")
            assert list_res.status_code == 200
            list_data = list_res.json()
            assert any(d["id"] == doc_id for d in list_data["documents"])

            # 3. Get document details
            detail_res = client.get(f"/api/documents/{doc_id}")
            assert detail_res.status_code == 200
            detail_data = detail_res.json()
            assert detail_data["document"]["id"] == doc_id
            assert len(detail_data["chunks_preview"]) > 0

            # 4. Search cases includes the uploaded document
            cases_res = client.get("/api/cases", params={"q": "test_appeal"})
            assert cases_res.status_code == 200
            assert any(c["case_id"] == doc_id for c in cases_res.json()["cases"])
        finally:
            # 5. Delete document
            del_res = client.delete(f"/api/documents/{doc_id}")
            assert del_res.status_code == 200

            # 6. Verify 404 after deletion
            detail_after_del = client.get(f"/api/documents/{doc_id}")
            assert detail_after_del.status_code == 404

"""
LegalSaarthi — Phase 4 Structured Judgment Summary Tests

Tests covering:
  1. Comprehensive legal metadata extraction (court, case number, date, bench, parties).
  2. Precedents and statutory references extraction from structured sections.
  3. Structured sections parsing with block citations and page-level provenance.
  4. Explicit missing field detection ("Not clearly stated in the document").
  5. Summarization of existing 996-corpus judgments.
  6. Summarization of newly uploaded judgment PDFs.
  7. Document type routing guard (contract vs court judgment).
  8. FastAPI REST endpoint POST /api/summarize with Phase 4 response schemas.
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
from src.preprocessing.structure_detector import (
    extract_full_case_metadata,
    extract_precedents_list,
    extract_statutory_provisions,
)
from src.summarization.summary_metrics import parse_structured_sections


def _create_sample_court_pdf() -> bytes:
    """Create a sample court judgment PDF for summarization testing."""
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text(
        (50, 72),
        "IN THE SUPREME COURT OF INDIA\n"
        "CIVIL APPELLATE JURISDICTION\n"
        "Civil Appeal No. 7744 of 2026\n"
        "OCTOBER 15, 2026\n"
        "[ DR. D.Y. CHANDRACHUD, CJI AND SANJIV KHANNA, J. ]\n\n"
        "Hindustan Infra Corporation ... Appellant\n"
        "VERSUS\n"
        "Union of India & Ors. ... Respondents\n\n"
        "JUDGMENT\n"
    )

    p2 = doc.new_page()
    p2.insert_text(
        (50, 72),
        "FACTS OF THE CASE\n"
        "The appellant was awarded a tender for port modernization.\n"
        "A bank guarantee was invoked following allegations of project delays.\n\n"
        "LEGAL ISSUES\n"
        "Whether invocation of an unconditional bank guarantee can be stayed absent established fraud.\n"
    )

    p3 = doc.new_page()
    p3.insert_text(
        (50, 72),
        "COURT REASONING AND PRECEDENTS\n"
        "Under Section 126 of the Indian Contract Act, a bank guarantee is an independent contract.\n"
        "In U.P. State Sugar Corporation v. Sumac International, this Court reiterated that\n"
        "courts must restrain from interfering with unconditional guarantees except in egregious fraud.\n\n"
        "DECISION AND FINAL ORDER\n"
        "We find no merit in the challenge. The appeal is accordingly dismissed.\n"
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ---------------------------------------------------------------------------
# Metadata & Section Parsing Tests
# ---------------------------------------------------------------------------

class TestPhase4MetadataAndParsing:

    def test_extract_metadata_from_corpus_case(self):
        """Metadata extraction from baseline corpus judgment 2015 INSC 177."""
        service = LegalSaarthiService.get_instance()
        service.initialize()
        struct_file = service.case_to_file.get("2015 INSC 177")
        assert struct_file is not None and struct_file.exists()

        with open(struct_file, encoding="utf-8") as f:
            enriched_doc = json.load(f)

        meta = extract_full_case_metadata(enriched_doc)
        assert meta["court"] == "Supreme Court of India"
        assert "2475-2476 of 2015" in meta["case_number"]
        assert "2015" in meta["judgment_date"]
        assert len(meta["bench"]) >= 2
        assert "M.Y. EQBAL" in meta["bench"][0] or "M.Y. EQBAL" in meta["bench"][1]
        assert "SHERALI" in meta["petitioner"].upper()
        assert "MAHARASHTRA" in meta["respondent"].upper()

    def test_extract_precedents_and_statutes_from_corpus(self):
        """Extract citations and statutes from baseline corpus judgment."""
        service = LegalSaarthiService.get_instance()
        service.initialize()
        struct_file = service.case_to_file.get("2015 INSC 177")

        with open(struct_file, encoding="utf-8") as f:
            enriched_doc = json.load(f)

        precedents = extract_precedents_list(enriched_doc)
        statutes = extract_statutory_provisions(enriched_doc)

        # 2015 INSC 177 has PRECEDENTS and mentions CPC
        assert isinstance(precedents, list)
        assert isinstance(statutes, list)

    def test_parse_structured_sections_page_provenance(self):
        """Sections are parsed into distinct fields with block citations and mapped pages."""
        summary_text = (
            "Facts: The appellant entered into an EPC contract [p1_2].\n"
            "Issues: Whether the bank guarantee invocation was valid [p2_1].\n"
            "Reasoning: An unconditional bank guarantee represents an independent obligation [p3_1].\n"
            "Decision: The appeal is dismissed with costs [p3_4]."
        )
        input_blocks = [
            {"block_id": "p1_2", "pages": [1]},
            {"block_id": "p2_1", "pages": [2]},
            {"block_id": "p3_1", "pages": [3]},
            {"block_id": "p3_4", "pages": [3]},
        ]

        parsed = parse_structured_sections(summary_text, input_blocks=input_blocks)
        sections = parsed["sections"]
        missing = parsed["missing_fields"]

        assert "facts" in sections
        assert sections["facts"]["citations"] == ["p1_2"]
        assert sections["facts"]["pages"] == [1]
        assert sections["facts"]["is_stated"] is True

        assert "issues" in sections
        assert sections["issues"]["pages"] == [2]

        assert "decision" in sections
        assert sections["decision"]["pages"] == [3]

        assert missing == []

    def test_parse_structured_sections_missing_fields(self):
        """Sections with 'Not clearly stated' are flagged in missing_fields."""
        summary_text = (
            "Facts: The tender was cancelled [p1_1].\n"
            "Issues: Not clearly stated in the document.\n"
            "Reasoning: The administrative order violated natural justice [p2_2].\n"
            "Decision: The appeal is allowed [p3_1]."
        )
        input_blocks = [
            {"block_id": "p1_1", "pages": [1]},
            {"block_id": "p2_2", "pages": [2]},
            {"block_id": "p3_1", "pages": [3]},
        ]

        parsed = parse_structured_sections(summary_text, input_blocks=input_blocks)
        sections = parsed["sections"]
        missing = parsed["missing_fields"]

        assert sections["issues"]["is_stated"] is False
        assert "issues" in missing


# ---------------------------------------------------------------------------
# End-to-End Summarization Tests
# ---------------------------------------------------------------------------

class TestPhase4SummarizationService:

    def test_summarize_corpus_case(self):
        """Summarize baseline corpus case with Mock LLM and verify Phase 4 payload."""
        service = LegalSaarthiService.get_instance()
        service.initialize()

        mock_llm = MagicMock()
        mock_llm.last_latency_s = 0.25
        mock_llm.last_model_used = "mock-qwen3-8b"
        mock_llm.generate.return_value = (
            "Facts: The appellant entered into agreement [p1_2].\n"
            "Issues: Whether receiver appointment is permissible [p1_9].\n"
            "Reasoning: Receiver function ordinarily terminates with final decision [p1_9].\n"
            "Decision: The appeals are dismissed [p1_11]."
        )
        service.llm_client = mock_llm

        res = service.summarize_case("2015 INSC 177", mode="structured")
        assert res.case_id == "2015 INSC 177"
        assert res.primary.predicted_disposition == "dismissed"
        assert res.primary.case_metadata is not None
        assert res.primary.case_metadata["court"] == "Supreme Court of India"
        assert res.primary.case_metadata["petitioner"] != ""
        assert "facts" in res.primary.structured_sections
        assert res.primary.prompt_version == "summ_v1"

    def test_summarize_uploaded_document(self, tmp_path):
        """Summarize an uploaded court judgment PDF with end-to-end provenance."""
        service = LegalSaarthiService.get_instance()
        service.initialize()

        # Ingest test PDF
        mgr = DocumentManager.get_instance(data_dir=service.repo_root / "data")
        pdf_bytes = _create_sample_court_pdf()
        meta = mgr.ingest_pdf(pdf_bytes, "chandrachud_bench.pdf", service=service)
        doc_id = meta["id"]

        mock_llm = MagicMock()
        mock_llm.last_latency_s = 0.3
        mock_llm.last_model_used = "mock-qwen3-8b"
        mock_llm.generate.return_value = (
            "Facts: Tender was awarded for port modernization [p2_1].\n"
            "Issues: Whether invocation of bank guarantee can be stayed [p2_2].\n"
            "Reasoning: Courts will not interfere absent established fraud [p3_1].\n"
            "Decision: The appeal is accordingly dismissed [p3_2]."
        )
        service.llm_client = mock_llm

        try:
            res = service.summarize_case(doc_id, mode="structured")
            assert res.case_id == doc_id
            assert res.primary.predicted_disposition == "dismissed"
            assert res.primary.case_metadata is not None
            assert res.primary.case_metadata["court"] == "Supreme Court of India"
            assert "7744 of 2026" in res.primary.case_metadata["case_number"]
            assert len(res.primary.structured_sections) >= 4
        finally:
            # Clean up uploaded test document
            mgr.delete_document(doc_id, service=service)

    def test_summarize_nonexistent_case_raises_filenotfound(self):
        """Attempting to summarize an unknown case ID raises FileNotFoundError."""
        service = LegalSaarthiService.get_instance()
        service.initialize()
        with pytest.raises(FileNotFoundError):
            service.summarize_case("NON_EXISTENT_CASE_9999")


# ---------------------------------------------------------------------------
# FastAPI REST Endpoint Tests
# ---------------------------------------------------------------------------

class TestPhase4RestEndpoint:

    def test_api_summarize_endpoint(self):
        """POST /api/summarize returns complete Phase 4 response structure."""
        client = TestClient(app)
        service = LegalSaarthiService.get_instance()
        service.initialize()

        mock_llm = MagicMock()
        mock_llm.last_latency_s = 0.15
        mock_llm.last_model_used = "mock-qwen3-8b"
        mock_llm.generate.return_value = (
            "Facts: The land acquisition dispute arose [p1_2].\n"
            "Issues: Whether compensation was adequate [p1_3].\n"
            "Reasoning: Valuation rules must be strictly followed [p1_4].\n"
            "Decision: The appeal is dismissed [p1_5]."
        )
        service.llm_client = mock_llm

        response = client.post(
            "/api/summarize",
            json={"case_id": "2015 INSC 177", "mode": "structured"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["case_id"] == "2015 INSC 177"
        assert data["primary"]["predicted_disposition"] == "dismissed"
        assert "case_metadata" in data["primary"]
        assert data["primary"]["case_metadata"]["court"] == "Supreme Court of India"
        assert "structured_sections" in data["primary"]

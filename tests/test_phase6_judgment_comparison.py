"""
Unit and integration tests for Phase 6 — Judgment Comparison.

Verifies:
- Comparison across key legal dimensions (facts/posture, issues, reasoning/authorities, outcome/order, precedent relation).
- Strict evidence preservation with page numbers and citations for both documents.
- Precedent treatment guardrail: no fabricated "followed" or "distinguished" claims without textual proof.
- Cross-comparison between corpus cases and uploaded documents.
- Error handling (404 / FileNotFoundError for non-existent documents).
- REST API endpoint POST /api/compare.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.models import CompareJudgmentsResponse, ComparisonDimensionItem
from src.api.service import LegalSaarthiService


@pytest.fixture(scope="module")
def service():
    """Initializes and returns the singleton service."""
    svc = LegalSaarthiService.get_instance()
    svc.initialize()
    return svc


@pytest.fixture(scope="module")
def client():
    """FastAPI test client."""
    return TestClient(app)


class TestPhase6JudgmentComparisonService:
    """Tests the business logic of judgment comparison in LegalSaarthiService."""

    def test_compare_two_corpus_cases(self, service: LegalSaarthiService):
        """Compares two standard Supreme Court cases from the corpus."""
        case_a = "2015 INSC 177"
        case_b = "2015 INSC 341"

        res: CompareJudgmentsResponse = service.compare_judgments(case_a, case_b)

        assert res.case_id_a == case_a
        assert res.case_id_b == case_b
        assert res.case_a_meta is not None
        assert res.case_b_meta is not None
        assert len(res.dimensions) == 5

        dim_keys = [d.dimension for d in res.dimensions]
        assert "facts_and_posture" in dim_keys
        assert "issues" in dim_keys
        assert "reasoning_and_authorities" in dim_keys
        assert "outcome_and_order" in dim_keys
        assert "precedent_treatment" in dim_keys

        # Check evidence preservation for facts
        dim_facts = next(d for d in res.dimensions if d.dimension == "facts_and_posture")
        assert len(dim_facts.doc_a_content) > 10
        assert len(dim_facts.doc_b_content) > 10
        assert dim_facts.synthesis != ""

        # Check outcome dimension
        dim_outcome = next(d for d in res.dimensions if d.dimension == "outcome_and_order")
        assert len(dim_outcome.doc_a_content) > 5
        assert len(dim_outcome.doc_b_content) > 5

        # Check precedent treatment relation
        assert res.precedent_relation in [
            "independent_authorities", "followed", "distinguished", "overruled", "cited"
        ]
        assert res.precedent_relation_evidence is not None

    def test_compare_nonexistent_case_raises_filenotfound(self, service: LegalSaarthiService):
        """Ensures an unknown case ID raises FileNotFoundError cleanly."""
        with pytest.raises(FileNotFoundError):
            service.compare_judgments("2015 INSC 177", "NONEXISTENT_CASE_XYZ_9999")

    def test_precedent_relation_guardrail(self, service: LegalSaarthiService):
        """Ensures we do NOT claim one case followed or distinguished another unless text explicitly says so."""
        res = service.compare_judgments("2015 INSC 177", "2015 INSC 341")
        if "2015 insc 341" not in service._resolve_case_data("2015 INSC 177")["full_text"].lower():
            assert res.precedent_relation == "independent_authorities"
            assert "No direct cross-citation detected" in res.precedent_relation_evidence

    def test_compare_uploaded_and_corpus_case(self, service: LegalSaarthiService, tmp_path: Path):
        """Tests comparing an uploaded document against a corpus case."""
        doc_id = "test_upload_comp_001"
        up_struct_dir = service.repo_root / "data" / "uploads" / "structured"
        up_struct_dir.mkdir(parents=True, exist_ok=True)
        up_file = up_struct_dir / f"{doc_id}.json"

        mock_payload = {
            "case_id": doc_id,
            "title": "M/s ABC Corp vs Commissioner of Taxes",
            "court": "High Court of Delhi",
            "year": "2024",
            "total_pages": 4,
            "paragraphs": [
                {
                    "para_id": "1",
                    "text": "HIGH COURT OF DELHI\nCivil Appeal No. 9988 of 2024\nM/s ABC Corp\nVERSUS\nCommissioner of Taxes\nDate: 12-04-2024",
                    "section": "HEADER",
                    "page": 1,
                    "pages": [1]
                },
                {
                    "para_id": "2",
                    "text": "The appellant is a registered company challenging tax assessment under Section 148 of the Income Tax Act.",
                    "section": "FACTS",
                    "page": 1,
                    "pages": [1]
                },
                {
                    "para_id": "3",
                    "text": "The primary issue is whether notice issued after statutory limitation period is valid in law.",
                    "section": "ISSUES",
                    "page": 2,
                    "pages": [2]
                },
                {
                    "para_id": "4",
                    "text": "The court held that mandatory limitation prescribed by Parliament cannot be extended administratively.",
                    "section": "REASONING",
                    "page": 3,
                    "pages": [3]
                },
                {
                    "para_id": "5",
                    "text": "For the reasons stated, the impugned assessment order is quashed and the appeal is allowed with costs.",
                    "section": "DECISION",
                    "page": 4,
                    "pages": [4]
                }
            ]
        }
        with open(up_file, "w", encoding="utf-8") as f:
            json.dump(mock_payload, f)

        try:
            res = service.compare_judgments(doc_id, "2015 INSC 177")
            assert res.case_id_a == doc_id
            assert res.case_id_b == "2015 INSC 177"
            assert len(res.dimensions) == 5
            dim_facts = next(d for d in res.dimensions if d.dimension == "facts_and_posture")
            assert "registered company" in dim_facts.doc_a_content
            assert 1 in dim_facts.doc_a_pages
        finally:
            if up_file.exists():
                up_file.unlink()


class TestPhase6RestEndpoints:
    """Tests the REST endpoint POST /api/compare."""

    def test_api_compare_success(self, client: TestClient):
        """Verifies POST /api/compare returns 200 with complete comparison response."""
        res = client.post(
            "/api/compare",
            json={
                "case_id_a": "2015 INSC 177",
                "case_id_b": "2015 INSC 341",
            }
        )
        assert res.status_code == 200
        data = res.json()
        assert data["case_id_a"] == "2015 INSC 177"
        assert data["case_id_b"] == "2015 INSC 341"
        assert len(data["dimensions"]) == 5
        assert "precedent_relation" in data
        assert "shared_statutes" in data
        assert "shared_precedents" in data

    def test_api_compare_404_for_unknown(self, client: TestClient):
        """Verifies POST /api/compare returns 404 when a document is not found."""
        res = client.post(
            "/api/compare",
            json={
                "case_id_a": "2015 INSC 177",
                "case_id_b": "UNKNOWN_CASE_9999",
            }
        )
        assert res.status_code == 404

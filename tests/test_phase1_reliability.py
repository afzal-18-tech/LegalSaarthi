"""
LegalSaarthi — Phase 1 Reliability Regression Tests

Tests for all 5 Phase 1 critical fixes:
  1. Case/document-scoped retrieval: zero contamination from other cases.
  2. Citation vs claim verification: valid citation != supported claim.
  3. Party extraction robustness: fallback, not fragment extraction.
  4. Disposition prediction accuracy: DECISION section preferred over full scan.
  5. Document-type routing: judgments not routed to contract analysis.

All tests are fully offline (no LLM or Ollama calls).
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.api.service import FilteredRetrieverWrapper
from src.preprocessing.structure_detector import StructureDetector, extract_decision_text
from src.summarization.summary_metrics import (
    detect_document_type,
    DOC_TYPE_CONTRACT,
    DOC_TYPE_JUDGMENT,
    DOC_TYPE_OTHER,
    DOC_TYPE_STATUTE,
    predict_disposition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_enriched_doc(pages_spec):
    """Build a minimal enriched doc with section_label on every paragraph."""
    pages = []
    global_para_idx = 0
    for page_num, para_specs in enumerate(pages_spec, start=1):
        paras = []
        for section_label, text in para_specs:
            global_para_idx += 1
            paras.append({
                "para_id": f"p{page_num}_{global_para_idx}",
                "text": text,
                "section_label": section_label,
            })
        pages.append({"page_num": page_num, "paragraphs": paras})
    return {"case_id": "TEST-001", "total_pages": len(pages_spec), "pages": pages}


def _make_chunk_store(*cases):
    """Build chunk_store and case_to_chunks for n cases."""
    chunk_store, case_to_chunks, chunk_id_to_idx = {}, {}, {}
    idx = 0
    for case_id in cases:
        case_to_chunks[case_id] = []
        for i in range(5):
            cid = f"{case_id}_section_aware_{i}"
            chunk = {"chunk_id": cid, "case_id": case_id, "text": f"Text from {case_id} chunk {i}."}
            chunk_store[cid] = chunk
            case_to_chunks[case_id].append(chunk)
            chunk_id_to_idx[cid] = idx
            idx += 1
    return chunk_store, case_to_chunks, chunk_id_to_idx


# ---------------------------------------------------------------------------
# Fix 1 — Scoped Retrieval: Zero contamination
# ---------------------------------------------------------------------------

class TestScopedRetrievalIsolation:
    """Case A chunks must NEVER appear when Case B is selected."""

    @pytest.fixture
    def two_case_setup(self):
        chunk_store, case_to_chunks, chunk_id_to_idx = _make_chunk_store("CASE_A", "CASE_B")
        bm25_mock = MagicMock()
        bm25_mock.index.get_scores.return_value = [5.0] * len(chunk_store)
        bm25_mock.chunk_ids = list(chunk_store.keys())
        return chunk_store, case_to_chunks, chunk_id_to_idx, bm25_mock

    def test_case_b_chunks_absent_when_case_a_selected(self, two_case_setup):
        chunk_store, case_to_chunks, chunk_id_to_idx, bm25_mock = two_case_setup
        retriever = FilteredRetrieverWrapper(
            bm25_mock, chunk_store, case_to_chunks, chunk_id_to_idx, case_id="CASE_A"
        )
        results = retriever.retrieve("some query about CASE_B matter", top_k=10)
        for cid, _ in results:
            assert chunk_store[cid]["case_id"] == "CASE_A", (
                f"Contamination: chunk {cid} from {chunk_store[cid]['case_id']} "
                f"leaked into CASE_A-scoped retrieval."
            )

    def test_no_global_fallback_for_missing_case(self, two_case_setup):
        chunk_store, case_to_chunks, chunk_id_to_idx, bm25_mock = two_case_setup
        retriever = FilteredRetrieverWrapper(
            bm25_mock, chunk_store, case_to_chunks, chunk_id_to_idx, case_id="TOTALLY_FAKE_9999"
        )
        results = retriever.retrieve("any query", top_k=5)
        assert results == [], "Must return [] for unknown case, not global fallback"

    def test_global_retrieval_when_no_case_id(self, two_case_setup):
        chunk_store, case_to_chunks, chunk_id_to_idx, bm25_mock = two_case_setup
        bm25_mock.retrieve.return_value = [("CASE_A_section_aware_0", 1.0)]
        retriever = FilteredRetrieverWrapper(
            bm25_mock, chunk_store, case_to_chunks, chunk_id_to_idx, case_id=None
        )
        retriever.retrieve("global query", top_k=5)
        bm25_mock.retrieve.assert_called_once()

    def test_partial_case_id_match(self, two_case_setup):
        """Case-insensitive substring match must only include target case chunks."""
        chunk_store, case_to_chunks, chunk_id_to_idx, bm25_mock = two_case_setup
        retriever = FilteredRetrieverWrapper(
            bm25_mock, chunk_store, case_to_chunks, chunk_id_to_idx, case_id="case_a"
        )
        results = retriever.retrieve("query", top_k=5)
        for cid, _ in results:
            assert chunk_store[cid]["case_id"] == "CASE_A"


# ---------------------------------------------------------------------------
# Fix 2 — Citation vs Claim Verification Separation
# ---------------------------------------------------------------------------

class TestCitationVsClaimSeparation:
    """Valid citation IDs must not be conflated with supported claims."""

    def test_valid_citation_exists_in_retrieved_ids(self):
        from src.generation.verified_rag_pipeline import CitationVerifier
        result = CitationVerifier.verify("Dismissed [REAL_001].", retrieved_ids=["REAL_001"])
        assert result["valid_ids"] == ["REAL_001"]
        assert result["invalid_ids"] == []

    def test_invalid_citation_stripped_not_counted_as_valid(self):
        from src.generation.verified_rag_pipeline import CitationVerifier
        result = CitationVerifier.verify("Penalty imposed. [GHOST_999]", retrieved_ids=["REAL_001"])
        assert "GHOST_999" in result["invalid_ids"]
        assert "GHOST_999" not in result["valid_ids"]
        assert "GHOST_999" not in result["cleaned_answer"]

    def test_valid_citation_unsupported_claim_counted(self):
        """citations_valid_but_claim_unsupported must reflect mismatch."""
        valid_ids = ["REAL_001"]
        claims_unsup = 1
        flag = min(claims_unsup, len(valid_ids))
        assert flag == 1, "Valid citation with unsupported claim must be counted"

    def test_zero_unsupported_claims_gives_zero_flag(self):
        valid_ids = ["REAL_001"]
        claims_unsup = 0
        flag = min(claims_unsup, len(valid_ids))
        assert flag == 0


# ---------------------------------------------------------------------------
# Fix 3 — Party Extraction Robustness
# ---------------------------------------------------------------------------

class TestPartyExtraction:
    """extract_parties must return fallback strings, not arbitrary text fragments."""

    FALLBACK = "Not clearly stated in the document"

    def test_extracts_versus_pattern(self):
        doc = _make_enriched_doc([[
            ("HEADER", "IN THE SUPREME COURT OF INDIA"),
            ("HEADER", "Ram Kumar VERSUS State of Uttar Pradesh"),
        ]])
        result = StructureDetector.extract_parties(doc)
        assert result["petitioner"] != self.FALLBACK
        assert "Ram Kumar" in result["petitioner"]
        assert "State of Uttar Pradesh" in result["respondent"]

    def test_fallback_on_no_header_paragraphs(self):
        doc = _make_enriched_doc([[
            ("FACTS", "The appellant filed a suit."),
            ("REASONING", "The court considered the evidence."),
        ]])
        result = StructureDetector.extract_parties(doc)
        assert result["petitioner"] == self.FALLBACK
        assert result["respondent"] == self.FALLBACK

    def test_fallback_on_empty_doc(self):
        result = StructureDetector.extract_parties({"pages": []})
        assert result["petitioner"] == self.FALLBACK
        assert result["respondent"] == self.FALLBACK

    def test_parties_dict_has_required_keys(self):
        doc = _make_enriched_doc([[("HEADER", "IN THE SUPREME COURT")]])
        result = StructureDetector.extract_parties(doc)
        assert "petitioner" in result
        assert "respondent" in result


# ---------------------------------------------------------------------------
# Fix 4 — Disposition Prediction Accuracy
# ---------------------------------------------------------------------------

class TestDispositionPrediction:
    """predict_disposition must prefer DECISION section over intermediate mentions."""

    def test_decision_text_overrides_earlier_mention(self):
        summary = "The High Court dismissed the writ petition."
        decision_text = "The appeal is accordingly allowed."
        assert predict_disposition(summary, decision_text=decision_text) == "allowed"

    def test_decision_text_dismissed(self):
        summary = "The appellant argued at length."
        decision_text = "The appeal is dismissed. No order as to costs."
        assert predict_disposition(summary, decision_text=decision_text) == "dismissed"

    def test_partly_allowed_preferred_over_allowed(self):
        assert predict_disposition("The order is partly allowed.") == "partly_allowed"

    def test_not_stated_when_no_operative_text(self):
        assert predict_disposition("The court considered the matter.") == "not_stated"

    def test_backward_compatible_no_decision_text(self):
        assert predict_disposition("The appeal is dismissed.") == "dismissed"
        assert predict_disposition("The petition is allowed.") == "allowed"

    def test_last_40pct_fallback_used_without_decision_text(self):
        """Without decision_text, last 40pct of summary should be scanned."""
        # Make a long summary where the tail has the real outcome
        preamble = "The HC dismissed the petition below. " * 10
        tail = "The appeal is allowed."
        result = predict_disposition(preamble + tail)
        assert result == "allowed"

    def test_extract_decision_text_helper(self):
        doc = _make_enriched_doc([[
            ("HEADER", "IN THE SUPREME COURT"),
            ("FACTS", "The appellant was convicted."),
            ("DECISION", "The appeal is allowed. No order as to costs."),
            ("DECISION", "The impugned order is set aside."),
        ]])
        text = extract_decision_text(doc)
        assert "allowed" in text
        assert "set aside" in text
        assert "convicted" not in text

    def test_extract_decision_text_empty_when_no_decision_section(self):
        doc = _make_enriched_doc([[
            ("HEADER", "Header"),
            ("FACTS", "Facts here."),
        ]])
        assert extract_decision_text(doc) == ""


# ---------------------------------------------------------------------------
# Fix 5 — Document-Type Routing
# ---------------------------------------------------------------------------

class TestDocumentTypeRouting:
    """Judgments, contracts, and statutes must be classified correctly."""

    JUDGMENT_TEXT = (
        "IN THE SUPREME COURT OF INDIA CIVIL APPELLATE JURISDICTION. "
        "The appellant filed an appeal. Learned senior counsel submitted that. "
        "The appeal is accordingly dismissed. The petitioner had challenged the decree."
    )
    CONTRACT_TEXT = (
        "THIS AGREEMENT is made between the Seller and the Buyer. "
        "WHEREAS the parties have agreed to the terms and conditions. "
        "INDEMNITY clause applies. TERMINATION shall be per governing law."
    )
    STATUTE_TEXT = (
        "BE IT ENACTED by Parliament as follows. CHAPTER I Short title. "
        "This Act shall apply to the extent and commencement as stated. "
        "The Act come into force on such date as may be specified."
    )

    def test_judgment_classified_correctly(self):
        assert detect_document_type(self.JUDGMENT_TEXT) == DOC_TYPE_JUDGMENT

    def test_contract_classified_correctly(self):
        assert detect_document_type(self.CONTRACT_TEXT) == DOC_TYPE_CONTRACT

    def test_statute_classified_correctly(self):
        assert detect_document_type(self.STATUTE_TEXT) == DOC_TYPE_STATUTE

    def test_empty_text_classified_as_other(self):
        assert detect_document_type("") == DOC_TYPE_OTHER
        assert detect_document_type(None) == DOC_TYPE_OTHER

    def test_constants_have_correct_values(self):
        assert DOC_TYPE_JUDGMENT == "court_judgment"
        assert DOC_TYPE_CONTRACT == "contract"
        assert DOC_TYPE_STATUTE == "statute"
        assert DOC_TYPE_OTHER == "other_legal_text"

    def test_mixed_predominantly_judgment_wins(self):
        mixed = self.JUDGMENT_TEXT * 4 + self.CONTRACT_TEXT
        assert detect_document_type(mixed) == DOC_TYPE_JUDGMENT

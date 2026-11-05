"""
LegalSaarthi — Structure Detector Tests (Phase 7)

Tests for the rule-based legal structure detector including
individual label classification, schema validation, positional
heuristics, and contextual continuity.
"""
import pytest

from src.preprocessing.structure_detector import SectionLabel, StructureDetector


@pytest.fixture
def detector():
    """Create a default StructureDetector instance."""
    return StructureDetector()


# ------------------------------------------------------------------
# Helper: Build a minimal parsed doc for detect_structure
# ------------------------------------------------------------------
def _make_parsed_doc(paragraphs_texts, case_id="TEST-001"):
    """Build a minimal parsed doc dict from a list of paragraph texts."""
    pages = []
    page_paras = []
    page_num = 1
    for i, text in enumerate(paragraphs_texts):
        page_paras.append({
            "para_id": f"p{page_num}_{len(page_paras) + 1}",
            "bbox": [0, 0, 100, 20],
            "text": text,
        })
        # Put ~5 paragraphs per page for realism
        if (i + 1) % 5 == 0:
            pages.append({"page_num": page_num, "paragraphs": page_paras})
            page_paras = []
            page_num += 1
    if page_paras:
        pages.append({"page_num": page_num, "paragraphs": page_paras})
    return {
        "case_id": case_id,
        "total_pages": page_num,
        "pages": pages,
        "parse_status": "success",
        "error_msg": "",
    }


# ==================================================================
# Individual label classification tests
# ==================================================================
class TestClassifyParagraph:
    """Test that known text patterns produce the expected section label."""

    def test_classify_header(self, detector):
        text = "IN THE SUPREME COURT OF INDIA CIVIL APPELLATE JURISDICTION"
        label = detector.classify_paragraph(text, 0.0, 1, None)
        assert label == SectionLabel.HEADER

    def test_classify_facts(self, detector):
        text = "The brief facts of the case are as follows. The appellant filed a suit."
        # Use mid-document position so HEADER bias doesn't interfere
        label = detector.classify_paragraph(text, 0.15, 3, None)
        assert label == SectionLabel.FACTS

    def test_classify_issues(self, detector):
        text = "The issues for determination in this appeal are as under:"
        label = detector.classify_paragraph(text, 0.20, 4, None)
        assert label == SectionLabel.ISSUES

    def test_classify_arguments(self, detector):
        text = "Learned senior counsel for the appellant submitted that the order is erroneous."
        label = detector.classify_paragraph(text, 0.30, 5, None)
        assert label == SectionLabel.ARGUMENTS

    def test_classify_law(self, detector):
        text = "Section 302 of the Indian Penal Code provides for punishment for murder."
        label = detector.classify_paragraph(text, 0.40, 6, None)
        assert label == SectionLabel.LAW

    def test_classify_precedents(self, detector):
        text = "This Court in (2010) 5 SCC 600 held that the principles are well established."
        label = detector.classify_paragraph(text, 0.50, 8, None)
        assert label == SectionLabel.PRECEDENTS

    def test_classify_reasoning(self, detector):
        text = "In our considered opinion, the High Court erred in its approach."
        label = detector.classify_paragraph(text, 0.65, 10, None)
        assert label == SectionLabel.REASONING

    def test_classify_decision(self, detector):
        text = "The appeal is accordingly dismissed. No order as to costs."
        label = detector.classify_paragraph(text, 0.95, 15, None)
        assert label == SectionLabel.DECISION

    def test_classify_unknown(self, detector):
        text = "Lorem ipsum dolor sit amet consectetur adipiscing elit."
        # Middle of document, no context, should be UNKNOWN
        label = detector.classify_paragraph(text, 0.50, 8, None)
        assert label == SectionLabel.UNKNOWN


# ==================================================================
# Enum completeness
# ==================================================================
class TestSectionLabelEnum:
    def test_section_label_enum_values(self):
        expected = {
            "HEADER", "FACTS", "ISSUES", "ARGUMENTS",
            "LAW", "PRECEDENTS", "REASONING", "DECISION", "UNKNOWN",
        }
        actual = {lbl.value for lbl in SectionLabel}
        assert actual == expected

    def test_section_label_count(self):
        assert len(SectionLabel) == 9


# ==================================================================
# Full document structure detection
# ==================================================================
class TestDetectStructure:
    def test_detect_structure_output_schema(self, detector):
        """Verify the output dict has the expected keys and structure."""
        doc = _make_parsed_doc([
            "IN THE SUPREME COURT OF INDIA",
            "CIVIL APPEAL NO. 1234 of 2020",
            "The facts of the case are as follows.",
            "Learned counsel submitted that the order is wrong.",
            "Section 302 IPC applies in this case.",
            "In (2010) 5 SCC 600 the Court held this principle.",
            "In our considered opinion the trial court was right.",
            "The appeal is accordingly dismissed.",
        ])
        result = detector.detect_structure(doc)

        # Top-level keys
        assert "case_id" in result
        assert "structure_version" in result
        assert result["structure_version"] == "v1.0-rules"
        assert "sections_summary" in result
        assert "pages" in result

        # sections_summary has all labels
        for lbl in SectionLabel:
            assert lbl.value in result["sections_summary"]

        # Every paragraph has section_label
        for page in result["pages"]:
            for para in page["paragraphs"]:
                assert "section_label" in para
                assert para["section_label"] in {lbl.value for lbl in SectionLabel}

    def test_detect_structure_empty_doc(self, detector):
        """An empty document should not crash and return zero counts."""
        doc = {"case_id": "EMPTY-001", "total_pages": 0, "pages": [], "parse_status": "success", "error_msg": ""}
        result = detector.detect_structure(doc)
        assert result["sections_summary"] == {lbl.value: 0 for lbl in SectionLabel}


# ==================================================================
# Positional heuristics
# ==================================================================
class TestPositionalHeuristics:
    def test_header_at_top(self, detector):
        """A generic short text on page 1 at position 0.0 should bias to HEADER."""
        text = "TANVI SARWAL"
        label = detector.classify_paragraph(text, 0.0, 1, None)
        assert label == SectionLabel.HEADER

    def test_decision_at_bottom(self, detector):
        """A decision-like text at position 0.95 should be DECISION."""
        text = "In the result, the appeal is dismissed."
        label = detector.classify_paragraph(text, 0.95, 20, None)
        assert label == SectionLabel.DECISION


# ==================================================================
# Contextual continuity
# ==================================================================
class TestContextualContinuity:
    def test_continuity_sustains_label(self, detector):
        """A bland paragraph following ARGUMENTS should stay ARGUMENTS."""
        text = "He further submitted that the entire approach was flawed and the lower court did not apply the correct test."
        # This has "submitted" which is ARGUMENTS, plus continuity
        label = detector.classify_paragraph(text, 0.35, 6, SectionLabel.ARGUMENTS)
        assert label == SectionLabel.ARGUMENTS

    def test_strong_pattern_overrides_continuity(self, detector):
        """A strong DECISION pattern should override ARGUMENTS continuity."""
        text = "The appeal is accordingly dismissed. No order as to costs."
        label = detector.classify_paragraph(text, 0.90, 18, SectionLabel.ARGUMENTS)
        assert label == SectionLabel.DECISION


# ==================================================================
# Precedent & Authority Extraction Normalization
# ==================================================================
class TestExtractPrecedents:
    def test_extract_valid_citations_and_reject_fragments(self):
        from src.preprocessing.structure_detector import extract_precedents_list

        dummy_doc = {
            "pages": [
                {
                    "page_num": 1,
                    "paragraphs": [
                        {
                            "section_label": "PRECEDENTS",
                            "text": (
                                "In Kesavananda Bharati v. State of Kerala, AIR 1973 SC 1461, this Court held that "
                                "the basic structure cannot be altered. Further in (2014) 5 SCC 438, the principle "
                                "was affirmed. The court held that the appellant had failed to prove damages v. the "
                                "state's contentions. Furthermore, in Maneka Gandhi v. Union of India, (1978) 1 SCC 248, "
                                "procedure established by law was clarified."
                            ),
                        }
                    ],
                }
            ]
        }

        precedents = extract_precedents_list(dummy_doc)
        # Valid citations should be captured
        assert any("AIR 1973 SC 1461" in p for p in precedents)
        assert any("Kesavananda Bharati v. State of Kerala" in p for p in precedents)
        assert any("Maneka Gandhi v. Union of India" in p for p in precedents)
        assert any("(1978) 1 SCC 248" in p for p in precedents)

        # Sentence fragments MUST NOT be captured as precedent
        for p in precedents:
            assert "The court held that" not in p
            assert "failed to prove damages" not in p
            assert "state's contentions" not in p


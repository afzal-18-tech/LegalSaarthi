"""Tests for src.preprocessing.pdf_parser module."""
import json
from pathlib import Path

import fitz
import pytest

from src.preprocessing.pdf_parser import PDFParser


@pytest.fixture
def dummy_pdf(tmp_path: Path) -> Path:
    """Create a temporary dummy PDF file with headers, footers, and text."""
    pdf_path = tmp_path / "dummy_test.pdf"
    
    # Create an empty document
    doc = fitz.open()
    
    # Add a page
    page = doc.new_page(width=612, height=792)  # US Letter
    
    # Add header (should be ignored by default parser)
    page.insert_text((50, 40), "Supreme Court of India (Header)", fontsize=10)
    
    # Add main text paragraph 1
    text1 = "This is the first paragraph.\nIt contains multiple lines\nbut should be joined."
    page.insert_text((50, 100), text1, fontsize=12)
    
    # Add main text paragraph 2
    text2 = "This is the second paragraph. It contains some legal facts."
    page.insert_text((50, 200), text2, fontsize=12)
    
    # Add footer (should be ignored by default parser)
    page.insert_text((50, 750), "Page 1 of 1 (Footer)", fontsize=10)
    
    # Save the document
    doc.save(str(pdf_path))
    doc.close()
    
    return pdf_path


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    """Create a temporary PDF file with no text (to simulate scanned image)."""
    pdf_path = tmp_path / "empty_test.pdf"
    doc = fitz.open()
    # Create a page and draw a rectangle (no text)
    page = doc.new_page(width=612, height=792)
    page.draw_rect(fitz.Rect(100, 100, 500, 700), color=(1, 0, 0), fill=(0, 1, 0))
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


class TestPDFParser:
    """Tests for PDFParser."""

    def test_parse_valid_pdf_with_header_footer_removal(self, dummy_pdf: Path):
        """Should extract text, join newlines, and remove header/footer."""
        parser = PDFParser(remove_header_footer=True)
        result = parser.parse_pdf(dummy_pdf, case_id="test_case_1")
        
        assert result["case_id"] == "test_case_1"
        assert result["total_pages"] == 1
        assert result["parse_status"] == "success"
        
        pages = result["pages"]
        assert len(pages) == 1
        
        paragraphs = pages[0]["paragraphs"]
        assert len(paragraphs) == 2  # Header and footer removed
        
        # Check text joining
        assert "This is the first paragraph. It contains multiple lines but should be joined." in paragraphs[0]["text"]
        assert "This is the second paragraph. It contains some legal facts." in paragraphs[1]["text"]
        
        # Check paragraph IDs
        assert paragraphs[0]["para_id"] == "p1_1"
        assert paragraphs[1]["para_id"] == "p1_2"

    def test_parse_valid_pdf_without_header_footer_removal(self, dummy_pdf: Path):
        """Should extract all text including header and footer."""
        parser = PDFParser(remove_header_footer=False)
        result = parser.parse_pdf(dummy_pdf, case_id="test_case_2")
        
        pages = result["pages"]
        paragraphs = pages[0]["paragraphs"]
        
        # Header + para1 + para2 + footer = 4 blocks
        assert len(paragraphs) == 4
        assert "Supreme Court of India (Header)" in paragraphs[0]["text"]
        assert "Page 1 of 1 (Footer)" in paragraphs[-1]["text"]

    def test_parse_empty_pdf(self, empty_pdf: Path):
        """Should detect lack of text and flag as possible scanned image."""
        parser = PDFParser()
        result = parser.parse_pdf(empty_pdf, case_id="empty_test_case")
        
        assert result["parse_status"] == "failed_possible_scan"
        assert "scanned image" in result["error_msg"].lower()
        
        # Still returns structure, just empty paragraphs
        assert len(result["pages"]) == 1
        assert len(result["pages"][0]["paragraphs"]) == 0

    def test_parse_missing_file(self):
        """Should raise FileNotFoundError for nonexistent file."""
        parser = PDFParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_pdf(Path("nonexistent_file.pdf"), case_id="missing_case")

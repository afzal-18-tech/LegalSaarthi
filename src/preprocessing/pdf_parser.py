"""
LegalSaarthi — PDF Parser (Phase 6)

Parses Indian Supreme Court PDFs using PyMuPDF (fitz) to extract text while
preserving page and paragraph boundaries. This forms the canonical text layer
for downstream chunking and structure detection.
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF

from src.utils.logging_setup import get_logger

logger = get_logger("preprocessing.parser")


class PDFParser:
    """Parses PDF documents into structured text blocks/paragraphs."""

    def __init__(self, remove_header_footer: bool = True):
        self.remove_header_footer = remove_header_footer

    def _clean_text(self, text: str) -> str:
        """Clean up extracted text."""
        # Replace multiple newlines with single newline or space
        # depending on context, but since this is a single block,
        # we'll just normalize whitespace for the paragraph.
        # However, we want to retain some structure, so let's just 
        # replace arbitrary newlines within a paragraph block with spaces,
        # and strip trailing whitespace.
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _is_header_footer(self, bbox: Tuple[float, float, float, float], page_height: float) -> bool:
        """
        Determine if a block is a header or footer based on its vertical position.
        bbox format: (x0, y0, x1, y1)
        """
        y0 = bbox[1]
        y1 = bbox[3]
        
        # Typically, headers are in the top 10% and footers in the bottom 10%
        # Or fixed points, say y1 < 50 for header, y0 > page_height - 50 for footer
        top_margin = 0.08 * page_height
        bottom_margin = 0.92 * page_height
        
        if y1 < top_margin:
            return True
        if y0 > bottom_margin:
            return True
            
        return False

    def parse_pdf(self, pdf_path: Path, case_id: str) -> Dict[str, Any]:
        """
        Parse a PDF file into a structured dictionary.

        Args:
            pdf_path: Path to the PDF file.
            case_id: The case ID associated with this document.

        Returns:
            Dictionary containing structured parsed text.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Failed to open PDF {pdf_path}: {e}")
            raise

        structured_data = {
            "case_id": case_id,
            "total_pages": len(doc),
            "pages": [],
            "parse_status": "success",
            "error_msg": ""
        }

        total_text_length = 0

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_height = page.rect.height
            
            # get_text("blocks") returns: (x0, y0, x1, y1, "lines in block", block_no, block_type)
            # block_type == 0 is text, block_type == 1 is image.
            blocks = page.get_text("blocks")
            
            paragraphs = []
            para_idx = 0
            
            # Sort blocks by y0 (top to bottom), then x0 (left to right)
            # PyMuPDF usually returns them in reading order, but sorting helps ensure it.
            # We sort slightly fuzzy on y to handle multi-column or misaligned text if needed,
            # but standard sorting usually works for single-column court docs.
            blocks.sort(key=lambda b: (b[1], b[0]))
            
            for block in blocks:
                # Check if it's a text block
                if block[6] == 0:
                    bbox = block[:4]
                    raw_text = block[4]
                    
                    if self.remove_header_footer and self._is_header_footer(bbox, page_height):
                        continue
                        
                    cleaned_text = self._clean_text(raw_text)
                    if cleaned_text:
                        paragraphs.append({
                            "para_id": f"p{page_num + 1}_{para_idx + 1}",
                            "bbox": [round(x, 2) for x in bbox],
                            "text": cleaned_text
                        })
                        para_idx += 1
                        total_text_length += len(cleaned_text)

            structured_data["pages"].append({
                "page_num": page_num + 1,
                "paragraphs": paragraphs
            })

        # If we extracted very little text from a multi-page document, it might be a scanned image.
        if len(doc) > 0 and total_text_length < len(doc) * 50:
            structured_data["parse_status"] = "failed_possible_scan"
            structured_data["error_msg"] = "Extracted very little text; document may be a scanned image."
            logger.warning(f"Case {case_id}: Possible scanned image detected (Total text: {total_text_length} chars, Pages: {len(doc)}).")

        doc.close()

        return structured_data

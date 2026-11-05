"""
LegalSaarthi — Document Chunker (Phase 8)

Converts structured JSON documents into retrievable chunk units.
Implements paragraph, fixed-size, and section-aware chunking strategies
while preserving provenance metadata (case_id, pages, paragraphs, sections).
"""
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union

from src.utils.logging_setup import get_logger

logger = get_logger("preprocessing.chunker")


@dataclass
class Chunk:
    """Represents a single retrievable text chunk with provenance."""
    chunk_id: str
    case_id: str
    pages: List[int]
    paragraphs: List[str]
    sections: List[str]
    text: str
    word_count: int
    source_reference: str
    strategy: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DocumentChunker:
    """
    Chunks structured documents according to specified strategies.
    Assumes the input document has been processed by StructureDetector.
    """

    def __init__(self, max_words: int = 200, overlap_words: int = 0):
        """
        Args:
            max_words: Target maximum words per chunk (soft limit for single paragraphs).
            overlap_words: Target word overlap between chunks (implemented at paragraph level).
        """
        self.max_words = max_words
        self.overlap_words = overlap_words

    def _get_word_count(self, text: str) -> int:
        return len(text.split())

    def _build_source_reference(self, case_id: str, paras: List[str]) -> str:
        """Create a human-readable citation for the chunk."""
        if not paras:
            return case_id
        if len(paras) == 1:
            return f"{case_id}, para {paras[0]}"
        return f"{case_id}, paras {paras[0]}-{paras[-1]}"

    def _flatten_paragraphs(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten nested pages/paragraphs into a single sequential list."""
        flat_paras = []
        for page in doc.get("pages", []):
            page_num = page.get("page_num", 0)
            for para in page.get("paragraphs", []):
                # Enrich paragraph with page_num for easier processing
                para_enriched = dict(para)
                para_enriched["page_num"] = page_num
                flat_paras.append(para_enriched)
        return flat_paras

    def chunk_by_paragraph(self, doc: Dict[str, Any]) -> List[Chunk]:
        """
        Strategy 1: One chunk per paragraph.
        """
        case_id = doc.get("case_id", "unknown")
        flat_paras = self._flatten_paragraphs(doc)
        chunks = []

        for idx, para in enumerate(flat_paras):
            text = para.get("text", "").strip()
            if not text:
                continue

            section = para.get("section_label", "UNKNOWN")
            para_id = para.get("para_id", f"unknown_{idx}")
            page_num = para.get("page_num", 0)

            chunk = Chunk(
                chunk_id=f"{case_id}_para_{idx}",
                case_id=case_id,
                pages=[page_num],
                paragraphs=[para_id],
                sections=[section],
                text=text,
                word_count=self._get_word_count(text),
                source_reference=self._build_source_reference(case_id, [para_id]),
                strategy="paragraph",
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk_from_buffer(
        self,
        case_id: str,
        buffer: List[Dict[str, Any]],
        chunk_index: int,
        strategy: str
    ) -> Chunk:
        """Helper to create a Chunk object from a buffer of paragraphs."""
        pages = sorted(list(set(p["page_num"] for p in buffer)))
        paragraphs = [p["para_id"] for p in buffer if "para_id" in p]
        
        # Preserve order of sections, remove consecutive duplicates
        sections = []
        for p in buffer:
            sec = p.get("section_label", "UNKNOWN")
            if not sections or sections[-1] != sec:
                sections.append(sec)

        text = " ".join(p.get("text", "").strip() for p in buffer)
        
        return Chunk(
            chunk_id=f"{case_id}_{strategy}_{chunk_index}",
            case_id=case_id,
            pages=pages,
            paragraphs=paragraphs,
            sections=sections,
            text=text,
            word_count=self._get_word_count(text),
            source_reference=self._build_source_reference(case_id, paragraphs),
            strategy=strategy,
        )

    def _group_paragraphs(
        self,
        flat_paras: List[Dict[str, Any]],
        case_id: str,
        strategy_name: str,
        section_aware: bool
    ) -> List[Chunk]:
        """
        Core grouping logic for fixed_size and section_aware strategies.
        Uses paragraph boundaries (soft limits).
        """
        chunks = []
        buffer = []
        current_words = 0
        chunk_idx = 0

        for para in flat_paras:
            text = para.get("text", "").strip()
            if not text:
                continue
                
            para_words = self._get_word_count(text)
            para_section = para.get("section_label", "UNKNOWN")

            # Check if we should flush the buffer
            flush_reason = False
            
            if buffer:
                # 1. Word limit reached
                if current_words + para_words > self.max_words:
                    flush_reason = True
                # 2. Section boundary crossed (if section_aware)
                elif section_aware and para_section != buffer[-1].get("section_label", "UNKNOWN"):
                    flush_reason = True

            if flush_reason:
                chunks.append(self._create_chunk_from_buffer(case_id, buffer, chunk_idx, strategy_name))
                chunk_idx += 1
                
                # Handle overlap: keep last N paragraphs that fit within overlap_words limit
                overlap_buffer = []
                overlap_words_count = 0
                for p in reversed(buffer):
                    p_words = self._get_word_count(p.get("text", ""))
                    if overlap_words_count + p_words <= self.overlap_words:
                        overlap_buffer.insert(0, p)
                        overlap_words_count += p_words
                    else:
                        break
                
                buffer = overlap_buffer
                current_words = overlap_words_count

            # Add current paragraph to buffer
            buffer.append(para)
            current_words += para_words

        # Flush remaining buffer
        if buffer:
            chunks.append(self._create_chunk_from_buffer(case_id, buffer, chunk_idx, strategy_name))

        return chunks

    def chunk_fixed_size(self, doc: Dict[str, Any]) -> List[Chunk]:
        """
        Strategy 2: Group paragraphs up to max_words, ignoring section boundaries.
        """
        case_id = doc.get("case_id", "unknown")
        flat_paras = self._flatten_paragraphs(doc)
        return self._group_paragraphs(flat_paras, case_id, "fixed_size", section_aware=False)

    def chunk_by_section(self, doc: Dict[str, Any]) -> List[Chunk]:
        """
        Strategy 3: Group paragraphs up to max_words, respecting section boundaries.
        A new chunk starts whenever the section label changes.
        """
        case_id = doc.get("case_id", "unknown")
        flat_paras = self._flatten_paragraphs(doc)
        return self._group_paragraphs(flat_paras, case_id, "section_aware", section_aware=True)

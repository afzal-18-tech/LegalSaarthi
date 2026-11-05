"""
LegalSaarthi — Document Chunker Tests (Phase 8)

Tests for the DocumentChunker: paragraph, fixed-size, and section-aware chunking.
"""
import pytest

from src.preprocessing.chunker import DocumentChunker, Chunk


@pytest.fixture
def sample_structured_doc():
    return {
        "case_id": "TEST_CASE_1",
        "total_pages": 2,
        "pages": [
            {
                "page_num": 1,
                "paragraphs": [
                    {
                        "para_id": "p1_1",
                        "text": "This is a short header paragraph.",
                        "section_label": "HEADER"
                    },
                    {
                        "para_id": "p1_2",
                        "text": "This is a slightly longer facts paragraph. " * 5,  # 8 words * 5 = 40 words
                        "section_label": "FACTS"
                    },
                    {
                        "para_id": "p1_3",
                        "text": "Another facts paragraph. " * 5,  # 3 words * 5 = 15 words
                        "section_label": "FACTS"
                    }
                ]
            },
            {
                "page_num": 2,
                "paragraphs": [
                    {
                        "para_id": "p2_1",
                        "text": "The law states clearly. " * 2, # 4 words * 2 = 8 words
                        "section_label": "LAW"
                    },
                    {
                        "para_id": "p2_2",
                        "text": "Therefore the appeal is allowed. " * 2, # 5 words * 2 = 10 words
                        "section_label": "DECISION"
                    }
                ]
            }
        ]
    }


def test_chunk_by_paragraph(sample_structured_doc):
    chunker = DocumentChunker()
    chunks = chunker.chunk_by_paragraph(sample_structured_doc)
    
    assert len(chunks) == 5
    assert chunks[0].strategy == "paragraph"
    assert chunks[0].paragraphs == ["p1_1"]
    assert chunks[0].sections == ["HEADER"]
    assert chunks[0].pages == [1]
    assert chunks[0].word_count == 6


def test_chunk_fixed_size(sample_structured_doc):
    # Set a low word limit so we can see grouping
    # Para word counts: p1_1(6), p1_2(35), p1_3(15), p2_1(8), p2_2(10)
    chunker = DocumentChunker(max_words=50)
    chunks = chunker.chunk_fixed_size(sample_structured_doc)
    
    # Expected chunks:
    # 1: p1_1(6) + p1_2(35) = 41 words
    # 2: p1_3(15) + p2_1(8) + p2_2(10) = 33 words
    
    assert len(chunks) == 2
    
    assert chunks[0].paragraphs == ["p1_1", "p1_2"]
    assert chunks[0].sections == ["HEADER", "FACTS"]
    assert chunks[0].pages == [1]
    assert chunks[0].word_count == 41
    
    assert chunks[1].paragraphs == ["p1_3", "p2_1", "p2_2"]
    assert chunks[1].sections == ["FACTS", "LAW", "DECISION"]
    assert chunks[1].pages == [1, 2]
    assert chunks[1].word_count == 33


def test_chunk_fixed_size_with_overlap(sample_structured_doc):
    # Para word counts: p1_1(6), p1_2(35), p1_3(15), p2_1(8), p2_2(10)
    # Set max_words to 60 so p1_1+p1_2+p1_3 = 56. Next is 8, 56+8=64 > 60.
    chunker = DocumentChunker(max_words=60, overlap_words=20)
    chunks = chunker.chunk_fixed_size(sample_structured_doc)
    
    # 1: p1_1(6) + p1_2(35) + p1_3(15) = 56 words.
    # Overlap logic: reversed buffer: p1_3(15) fits in 20. overlap buffer = [p1_3].
    # Next para p1_2(35) + 15 = 50 > 20, breaks.
    # So overlap buffer has p1_3.
    # 2: starts with overlap buffer p1_3(15) + p2_1(8) + p2_2(10) = 33 words.
    
    assert len(chunks) == 2
    assert chunks[0].paragraphs == ["p1_1", "p1_2", "p1_3"]
    assert chunks[1].paragraphs == ["p1_3", "p2_1", "p2_2"]


def test_chunk_by_section(sample_structured_doc):
    # Para word counts: p1_1(6)[HEADER], p1_2(40)[FACTS], p1_3(15)[FACTS], p2_1(8)[LAW], p2_2(10)[DECISION]
    chunker = DocumentChunker(max_words=100) # large limit, should only split on sections
    chunks = chunker.chunk_by_section(sample_structured_doc)
    
    # Expected chunks:
    # 1: p1_1 (HEADER)
    # 2: p1_2, p1_3 (FACTS)
    # 3: p2_1 (LAW)
    # 4: p2_2 (DECISION)
    
    assert len(chunks) == 4
    
    assert chunks[0].sections == ["HEADER"]
    assert chunks[1].sections == ["FACTS"]
    assert chunks[1].paragraphs == ["p1_2", "p1_3"]
    assert chunks[2].sections == ["LAW"]
    assert chunks[3].sections == ["DECISION"]
    
def test_chunk_by_section_with_max_words(sample_structured_doc):
    # Para word counts: p1_1(6)[HEADER], p1_2(40)[FACTS], p1_3(15)[FACTS], p2_1(8)[LAW], p2_2(10)[DECISION]
    chunker = DocumentChunker(max_words=45) 
    chunks = chunker.chunk_by_section(sample_structured_doc)
    
    # Expected chunks:
    # 1: p1_1 (HEADER)
    # 2: p1_2 (FACTS) - 40 words. Next is p1_3 (15). 40+15=55 > 45, so flush.
    # 3: p1_3 (FACTS)
    # 4: p2_1 (LAW)
    # 5: p2_2 (DECISION)
    
    assert len(chunks) == 5
    assert chunks[1].paragraphs == ["p1_2"]
    assert chunks[2].paragraphs == ["p1_3"]


def test_source_reference():
    chunker = DocumentChunker()
    # 1 para
    assert chunker._build_source_reference("TEST", ["p1_1"]) == "TEST, para p1_1"
    # > 1 para
    assert chunker._build_source_reference("TEST", ["p1_1", "p1_2", "p2_1"]) == "TEST, paras p1_1-p2_1"
    # 0 paras
    assert chunker._build_source_reference("TEST", []) == "TEST"

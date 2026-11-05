"""
LegalSaarthi — API Data Models & Schemas (Phase 24)

Defines Pydantic models for request/response serialization across:
- Case search & metadata retrieval
- Question Answering (Generic LLM, Basic RAG, Verified-RAG, Comparison)
- Judgment Summarization (Structured & Plain-Language)
- Chunk Provenance Inspector
- Research Metrics & System Health
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CitationItem(BaseModel):
    """Provenance record for a single cited evidence chunk."""
    chunk_id: str
    case_id: str
    sections: List[str] = Field(default_factory=list)
    pages: List[int] = Field(default_factory=list)
    paragraphs: List[str] = Field(default_factory=list)
    text: str = ""
    source_reference: str = ""
    is_valid: bool = True
    is_verified: bool = True
    case_matches_scope: bool = True
    # Phase 1: Explicitly separate citation existence from claim-level entailment.
    # A citation can be valid (chunk exists and was retrieved) yet the claim it
    # supports may still be unsupported according to the claim verifier.
    claim_entailed: Optional[bool] = Field(
        default=None,
        description="True if the claim using this citation was entailed by evidence, "
                    "False if unsupported, None if not yet verified.",
    )


class QASystemResult(BaseModel):
    """Output from a single QA system (Generic, Basic RAG, or Verified-RAG)."""
    system: str
    system_name: str
    answer: str
    abstained: bool = False
    abstention_reason: Optional[str] = None
    cited_chunks: List[CitationItem] = Field(default_factory=list)
    citations_stripped_count: int = 0
    mean_claim_support: Optional[float] = None
    claims_supported: int = 0
    claims_total: int = 0
    statutory_references: List[str] = Field(default_factory=list)
    retrieval_scope: str = "global"
    case_fidelity_passed: bool = True
    mismatched_citation_ids: List[str] = Field(default_factory=list)
    citation_precision_score: Optional[float] = None
    claim_grounding_score: Optional[float] = None
    latency_s: float = 0.0
    # Phase 1: count of citations that exist but whose claims were unsupported
    citations_valid_but_claim_unsupported: int = Field(
        default=0,
        description="Number of citations that are lexically valid (chunk exists) but "
                    "whose associated claim was not entailed by the cited evidence. "
                    "Non-zero means the system has valid citations with weak grounding.",
    )


class QARequest(BaseModel):
    """Request payload for legal question answering."""
    query: str = Field(..., min_length=3, description="Legal question to ask")
    system: str = Field("verified_rag", description="'verified_rag', 'basic_rag', 'generic_llm', or 'compare_all'")
    case_id: Optional[str] = Field(None, description="Optional filter to a specific Supreme Court case ID")
    top_k: int = Field(5, ge=1, le=15, description="Number of evidence chunks to retrieve")


class QAResponse(BaseModel):
    """Response payload for legal question answering."""
    query: str
    case_id_filter: Optional[str] = None
    retrieval_scope: str = "global"
    primary: QASystemResult
    comparison: Optional[Dict[str, QASystemResult]] = None


class BenchmarkPromptItem(BaseModel):
    """Curated gold benchmark question with provenance and intended target."""
    query_id: Optional[str] = None
    tier: str
    case_id: Optional[str] = None
    case_title: Optional[str] = None
    prompt: str
    gold_evidence: List[str] = Field(default_factory=list)
    retrieval_scope: str = "case_scoped"


class SummaryResult(BaseModel):
    """Summary result with readability metrics and section provenance."""
    mode: str
    mode_name: str
    case_id: str
    summary: str
    flesch_reading_ease: Optional[float] = None
    fk_grade_level: Optional[float] = None
    word_count: int = 0
    predicted_disposition: str = "not_stated"
    citations: List[str] = Field(default_factory=list)
    latency_s: float = 0.0
    # Phase 1: document type routing guard
    document_type: str = Field(
        default="court_judgment",
        description="Detected document type: court_judgment | contract | statute | other_legal_text."
    )
    # Phase 1: extracted parties (fallback to 'Not clearly stated' if unavailable)
    parties: Dict[str, str] = Field(
        default_factory=dict,
        description="Extracted petitioner and respondent names, or 'Not clearly stated in the document'.",
    )
    # Phase 4: Reusable structured summary fields & provenance
    case_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extracted case metadata: court, case_number, judgment_date, bench, parties.",
    )
    structured_sections: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Parsed substantive sections (facts, issues, arguments, reasoning, decision, relief) with block citations and page numbers.",
    )
    cited_precedents: List[str] = Field(
        default_factory=list,
        description="Precedents cited in the judgment.",
    )
    statutory_provisions: List[str] = Field(
        default_factory=list,
        description="Statutes and provisions cited in the judgment.",
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="Fields explicitly identified as not clearly stated in the document.",
    )
    generated_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
        description="Timestamp when the summary was generated.",
    )
    prompt_version: str = Field(
        default="summ_v2",
        description="Version of the summarization prompt used.",
    )


class SummarizeRequest(BaseModel):
    """Request payload for judgment summarization."""
    case_id: str = Field(..., description="Indian Supreme Court case identifier (e.g. '2015 INSC 177')")
    mode: str = Field("structured", description="'structured', 'plain_language', or 'compare'")


class SummarizeResponse(BaseModel):
    """Response payload for judgment summarization."""
    case_id: str
    case_title: Optional[str] = None
    year: Optional[str] = None
    split: Optional[str] = None
    primary: SummaryResult
    comparison: Optional[SummaryResult] = None


class CaseListItem(BaseModel):
    """Summary of a case for search and browsing."""
    case_id: str
    case_title: Optional[str] = None
    year: str = ""
    split: str = "dev"
    court: str = "Supreme Court of India"
    total_paragraphs: int = 0
    total_pages: int = 0
    sections_summary: Dict[str, int] = Field(default_factory=dict)


class CaseSearchResponse(BaseModel):
    """Paginated list of Supreme Court cases."""
    total: int
    page: int
    page_size: int
    cases: List[CaseListItem]


class CaseDetailResponse(BaseModel):
    """Detailed case metadata and structural paragraphs."""
    case_id: str
    case_title: Optional[str] = None
    year: str
    split: str
    court: str
    total_pages: int
    total_paragraphs: int
    sections_summary: Dict[str, int]
    source_url: Optional[str] = None
    sample_paragraphs: List[Dict[str, Any]] = Field(default_factory=list)


class ChunkProvenanceResponse(BaseModel):
    """Full detail of a single evidence chunk."""
    chunk_id: str
    case_id: str
    sections: List[str]
    pages: List[int]
    paragraphs: List[str]
    text: str
    word_count: int
    source_reference: str
    strategy: str = "section_aware"


class SystemHealthResponse(BaseModel):
    """System status and loaded component counts."""
    status: str = "ok"
    version: str = "0.1.0"
    model_name: str = "ollama/qwen3:8b"
    llm_connected: bool = True
    total_cases_indexed: int = 0
    total_chunks_loaded: int = 0
    bm25_index_loaded: bool = False
    device: str = "cpu"


class ResearchMetricsResponse(BaseModel):
    """Key scientific benchmark and evaluation metrics across phases."""
    research_question: str
    hypotheses: Dict[str, str]
    baselines: Dict[str, Dict[str, Any]]
    judge_agreement: Dict[str, Any]
    corpus_stats: Dict[str, Any]


# ---------------------------------------------------------------------------
# Phase 2 — Uploaded Document Models
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    """Metadata schema for an uploaded court judgment or legal document."""
    id: str
    original_filename: str
    stored_path: str
    file_hash: str
    document_type: str = "court_judgment"
    upload_time: str
    status: str = "uploaded"  # uploaded, extracting, chunking, indexing, ready, failed
    page_count: int = 0
    chunk_count: int = 0
    extraction_method: str = "pymupdf_fitz"
    error_message: Optional[str] = None
    pipeline_version: str = "1.0.0"
    per_page_text_lengths: Dict[str, int] = Field(default_factory=dict)
    parties: Optional[Dict[str, Any]] = None


class DocumentUploadResponse(BaseModel):
    """Response returned upon uploading or processing a document."""
    document: DocumentMetadata
    message: str
    status: str


class DocumentListResponse(BaseModel):
    """Response listing uploaded documents."""
    documents: List[DocumentMetadata]
    total: int


class DocumentDetailResponse(BaseModel):
    """Detailed view of an uploaded document including its extracted chunks."""
    document: DocumentMetadata
    chunks_preview: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 5 — In-Document Search & Lightweight Query History Models
# ---------------------------------------------------------------------------

class DocumentSearchHit(BaseModel):
    """Individual match from in-document keyword/phrase search."""
    chunk_id: str
    document_id: str
    document_name: str = ""
    page_start: int = 1
    page_end: int = 1
    pages: List[int] = Field(default_factory=list)
    section: str = "UNKNOWN"
    score: float
    match_type: str = "keyword"  # "keyword", "phrase", "semantic"
    snippet: str
    source_reference: str


class DocumentSearchResponse(BaseModel):
    """Results of searching within a single uploaded document."""
    document_id: str
    query: str
    total_hits: int
    hits: List[DocumentSearchHit]


class QueryHistoryItem(BaseModel):
    """Single recorded question answering session event."""
    query_id: str
    timestamp: str
    query: str
    case_id: Optional[str] = None
    document_name: str
    retrieval_scope: str = "global"
    system: str = "verified_rag"
    answer_preview: str = ""
    abstained: bool = False
    top_citation: Optional[str] = None
    latency_s: float = 0.0


class QueryHistoryResponse(BaseModel):
    """List of recent queries."""
    total: int
    history: List[QueryHistoryItem]


# ── Phase 6: Judgment Comparison Models ────────────────────────────────────

class ComparisonDimensionItem(BaseModel):
    """Comparison evaluation for a specific legal dimension between two judgments."""
    dimension: str = Field(..., description="Key: facts_and_posture | issues | reasoning_and_authorities | outcome_and_order | precedent_treatment")
    dimension_title: str
    doc_a_content: str
    doc_a_pages: List[int] = Field(default_factory=list)
    doc_a_citations: List[str] = Field(default_factory=list)
    doc_b_content: str
    doc_b_pages: List[int] = Field(default_factory=list)
    doc_b_citations: List[str] = Field(default_factory=list)
    synthesis: str = Field(..., description="Comparative analysis highlighting alignment, distinctions, or differences.")
    comparable: bool = True


class CompareJudgmentsRequest(BaseModel):
    """Request payload for comparing two judgments."""
    case_id_a: str = Field(..., description="Primary case identifier or uploaded document ID.")
    case_id_b: str = Field(..., description="Secondary case identifier or uploaded document ID.")


class CompareJudgmentsResponse(BaseModel):
    """Structured side-by-side comparison between two judgments."""
    case_id_a: str
    case_id_b: str
    case_a_meta: Dict[str, Any] = Field(default_factory=dict)
    case_b_meta: Dict[str, Any] = Field(default_factory=dict)
    dimensions: List[ComparisonDimensionItem] = Field(default_factory=list)
    precedent_relation: str = Field(
        default="independent_authorities",
        description="Relationship: followed | distinguished | overruled | cited | independent_authorities | not_applicable"
    )
    precedent_relation_evidence: Optional[str] = Field(
        default=None,
        description="Textual excerpt proving cross-citation treatment, or explanation if independent."
    )
    shared_statutes: List[str] = Field(default_factory=list)
    shared_precedents: List[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    latency_s: float = 0.0




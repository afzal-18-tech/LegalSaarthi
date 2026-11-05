"""
LegalSaarthi — FastAPI Interactive Application (Phase 24)

Serves:
- REST API for Question Answering, Summarization, and Provenance Inspection
- Static frontend UI assets (Glassmorphic dark design system)
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from src.api.auth import AuthManager, UserSignupRequest, UserLoginRequest, AuthResponse, UserProfile, UserRole
from src.api.models import (
    BenchmarkPromptItem,
    CaseDetailResponse,
    CaseSearchResponse,
    ChunkProvenanceResponse,
    CompareJudgmentsRequest,
    CompareJudgmentsResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentSearchHit,
    DocumentSearchResponse,
    DocumentUploadResponse,
    QARequest,
    QAResponse,
    QueryHistoryItem,
    QueryHistoryResponse,
    ResearchMetricsResponse,
    SummarizeRequest,
    SummarizeResponse,
    SystemHealthResponse,
)
from src.api.service import LegalSaarthiService

app = FastAPI(
    title="LegalSaarthi API",
    description="Evidence-Grounded AI for Indian Court Judgment Summarization and Question Answering",
    version="0.1.0",
)

# CORS middleware for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Silence browser favicon 404 logs."""
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def serve_root():
    """Serve the single-page application with no-cache headers."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(
            str(index_file),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return HTMLResponse("<h2>LegalSaarthi is running. Static index.html not yet generated.</h2>")


@app.get("/api/health", response_model=SystemHealthResponse)
def get_health():
    """Return system readiness, component status, and loaded chunk statistics."""
    service = LegalSaarthiService.get_instance()
    return service.check_health()


# ---------------------------------------------------------------------------
# Authentication & User Role Endpoints (LAWYER, RESEARCHER, STUDENT)
# ---------------------------------------------------------------------------

@app.post("/api/auth/signup", response_model=AuthResponse)
def signup(req: UserSignupRequest):
    """Register a new user account with role: LAWYER, RESEARCHER, or STUDENT."""
    auth_mgr = AuthManager.get_instance()
    try:
        return auth_mgr.signup(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@app.post("/api/auth/login", response_model=AuthResponse)
def login(req: UserLoginRequest):
    """Authenticate user with email and password."""
    auth_mgr = AuthManager.get_instance()
    try:
        return auth_mgr.login(req)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


@app.get("/api/auth/me", response_model=UserProfile)
def get_current_user(token: Optional[str] = Query(None), authorization: Optional[str] = Header(None)):
    """Retrieve verified user profile from bearer token or query parameter."""
    auth_mgr = AuthManager.get_instance()
    auth_token = token
    if not auth_token and authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            auth_token = parts[1]

    if not auth_token:
        raise HTTPException(status_code=401, detail="Authentication token required.")

    profile = auth_mgr.verify_token(auth_token)
    if not profile:
        raise HTTPException(status_code=401, detail="Session expired or invalid token.")
    return profile


@app.post("/api/auth/logout")
def logout(token: Optional[str] = Query(None), authorization: Optional[str] = Header(None)):
    """Logout and revoke the session token."""
    auth_mgr = AuthManager.get_instance()
    auth_token = token
    if not auth_token and authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            auth_token = parts[1]
    if auth_token:
        auth_mgr.logout(auth_token)
    return {"status": "ok", "message": "Logged out successfully."}


@app.get("/api/auth/demo-accounts")
def get_demo_accounts():
    """Return pre-configured demo credentials for 1-click evaluation across LAWYER, RESEARCHER, and STUDENT."""
    return [
        {
            "role": "LAWYER",
            "title": "Advocate / Legal Counsel",
            "name": "Adv. Rajesh Sharma",
            "email": "advocate.sharma@legal.in",
            "password": "Password@123",
            "organization": "Supreme Court Bar Association",
        },
        {
            "role": "RESEARCHER",
            "title": "Academic / AI Researcher",
            "name": "Dr. Sunita Verma",
            "email": "dr.verma@research.ac.in",
            "password": "Password@123",
            "organization": "National Law University, Delhi",
        },
        {
            "role": "STUDENT",
            "title": "Law / AI-ML Student",
            "name": "Aryan Patel",
            "email": "aryan.mca@university.edu",
            "password": "Password@123",
            "organization": "Department of Computer Applications (MCA)",
        },
    ]


@app.get("/api/cases", response_model=CaseSearchResponse)
def search_cases(
    q: Optional[str] = Query(None, description="Search query in case ID or title"),
    year: Optional[str] = Query(None, description="Filter by year (e.g. '2015')"),
    split: Optional[str] = Query(None, description="Filter by split ('dev', 'val', 'test')"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=1500, description="Page size"),
):
    """Search and browse Indian Supreme Court judgments."""
    service = LegalSaarthiService.get_instance()
    return service.search_cases(q=q, year=year, split=split, page=page, page_size=page_size)


@app.get("/api/cases/all", response_model=CaseSearchResponse)
def get_all_cases():
    """Retrieve all 993 Supreme Court judgments in a single fast call for dropdowns."""
    service = LegalSaarthiService.get_instance()
    return service.search_cases(page=1, page_size=1500)


@app.get("/api/cases/{case_id:path}", response_model=CaseDetailResponse)
def get_case_detail(case_id: str):
    """Get metadata and section paragraphs for a single judgment."""
    service = LegalSaarthiService.get_instance()
    res = service.get_case_details(case_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found.")
    return res


@app.get("/api/chunks/{chunk_id:path}", response_model=ChunkProvenanceResponse)
def get_chunk(chunk_id: str):
    """Retrieve full text and provenance for a specific evidence chunk."""
    service = LegalSaarthiService.get_instance()
    res = service.get_chunk_provenance(chunk_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found.")
    return res


@app.post("/api/qa", response_model=QAResponse)
def answer_question(req: QARequest):
    """Answer a legal question using Verified-RAG, Basic RAG, or Generic LLM."""
    service = LegalSaarthiService.get_instance()
    try:
        return service.answer_query(
            query=req.query,
            system=req.system,
            case_id=req.case_id,
            top_k=req.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize_case(req: SummarizeRequest):
    """Generate structured or plain-language summaries with readability metrics."""
    service = LegalSaarthiService.get_instance()
    try:
        return service.summarize_case(case_id=req.case_id, mode=req.mode)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare", response_model=CompareJudgmentsResponse)
def compare_judgments(req: CompareJudgmentsRequest):
    """Compare two court judgments across facts, issues, reasoning, outcome, and precedents."""
    service = LegalSaarthiService.get_instance()
    try:
        return service.compare_judgments(case_id_a=req.case_id_a, case_id_b=req.case_id_b)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Phase 2 — Uploaded Document Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
@app.post("/api/documents", response_model=DocumentUploadResponse)
def upload_document(file: UploadFile = File(...)):
    """Upload a previously unseen court-judgment PDF and run ingestion pipeline."""
    service = LegalSaarthiService.get_instance()
    try:
        content = file.file.read()
        filename = file.filename or "uploaded.pdf"
        doc = service.upload_document(content=content, original_filename=filename)
        status_msg = (
            "Document successfully processed and ready for scoped Q&A."
            if doc.get("status") == "ready"
            else f"Document ingestion failed: {doc.get('error_message')}"
        )
        return DocumentUploadResponse(
            document=DocumentMetadata(**doc),
            message=status_msg,
            status=doc.get("status", "unknown"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {e}")


@app.get("/api/documents", response_model=DocumentListResponse)
def list_documents():
    """List all uploaded judgments with current status and page/chunk counts."""
    service = LegalSaarthiService.get_instance()
    docs = service.list_uploaded_documents()
    return DocumentListResponse(
        documents=[DocumentMetadata(**d) for d in docs],
        total=len(docs),
    )


@app.get("/api/documents/{doc_id}", response_model=DocumentDetailResponse)
def get_document(doc_id: str):
    """Retrieve details and extracted chunks preview for an uploaded document."""
    service = LegalSaarthiService.get_instance()
    doc = service.get_uploaded_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Uploaded document '{doc_id}' not found.")
    chunks = service.get_uploaded_document_chunks(doc_id)
    return DocumentDetailResponse(
        document=DocumentMetadata(**doc),
        chunks_preview=chunks[:10],
    )


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete an uploaded document and its chunks without touching the 996-corpus."""
    service = LegalSaarthiService.get_instance()
    deleted = service.delete_uploaded_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Uploaded document '{doc_id}' not found.")
    return {"status": "ok", "message": f"Document '{doc_id}' deleted successfully."}


@app.post("/api/documents/{doc_id}/retry", response_model=DocumentUploadResponse)
def retry_document(doc_id: str):
    """Retry ingestion for an uploaded document whose processing failed."""
    service = LegalSaarthiService.get_instance()
    try:
        doc = service.retry_uploaded_document(doc_id)
        return DocumentUploadResponse(
            document=DocumentMetadata(**doc),
            message=f"Document reprocessed: status is '{doc.get('status')}'.",
            status=doc.get("status", "unknown"),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retry failed: {e}")


# ---------------------------------------------------------------------------
# Phase 5 — In-Document Search & Lightweight History Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/documents/{doc_id}/search", response_model=DocumentSearchResponse)
def search_in_document(
    doc_id: str,
    q: str = Query(..., min_length=1, description="Keywords or exact phrase to search in document"),
    top_k: int = Query(10, ge=1, le=50, description="Max matches to return"),
):
    """Keyword and exact phrase search within the chunks of a single uploaded document."""
    service = LegalSaarthiService.get_instance()
    doc = service.get_uploaded_document(doc_id)
    norm_id = doc_id.replace("_", " ") if "_" in doc_id else doc_id
    if not doc and doc_id not in service.cases_meta and norm_id not in service.cases_meta and doc_id not in service.case_to_chunks and norm_id not in service.case_to_chunks:
        raise HTTPException(status_code=404, detail=f"Document or case '{doc_id}' not found.")

    hits = service.search_in_document(doc_id, query=q, top_k=top_k)
    return DocumentSearchResponse(
        document_id=doc_id,
        query=q,
        total_hits=len(hits),
        hits=[DocumentSearchHit(**h) for h in hits],
    )


@app.get("/api/history", response_model=QueryHistoryResponse)
def get_query_history(
    case_id: Optional[str] = Query(None, description="Filter history by case or document ID"),
    limit: int = Query(50, ge=1, le=200, description="Max history records to return"),
):
    """Retrieve private local research query history."""
    service = LegalSaarthiService.get_instance()
    records = service.list_history(case_id=case_id, limit=limit)
    return QueryHistoryResponse(
        total=len(records),
        history=[QueryHistoryItem(**r) for r in records],
    )


@app.delete("/api/history/{query_id}")
def delete_history_item(query_id: str):
    """Delete a single query history entry."""
    service = LegalSaarthiService.get_instance()
    deleted = service.delete_history_item(query_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Query history record '{query_id}' not found.")
    return {"status": "ok", "message": f"Query '{query_id}' deleted."}


@app.delete("/api/history")
def clear_query_history():
    """Clear all local query history records."""
    service = LegalSaarthiService.get_instance()
    count = service.clear_history()
    return {"status": "ok", "message": f"Cleared {count} query history records."}




@app.get("/api/metrics", response_model=ResearchMetricsResponse)
def get_metrics():
    """Return scientific baseline metrics and LLM judge agreement outcomes."""
    service = LegalSaarthiService.get_instance()
    return service.get_research_metrics()


@app.get("/api/benchmark-prompts", response_model=List[BenchmarkPromptItem])
def get_benchmark_prompts() -> List[BenchmarkPromptItem]:
    """Return universal gold legal queries supported across all court judgments and uploaded PDFs."""
    return [
        BenchmarkPromptItem(
            query_id="GOLD_001",
            tier="Final Order / Disposition",
            case_id=None,
            case_title="Universal Legal Query",
            prompt="What was the final order or disposition of the Supreme Court in this appeal?",
            gold_evidence=[],
            retrieval_scope="case_scoped",
        ),
        BenchmarkPromptItem(
            query_id="GOLD_002",
            tier="Facts & Procedural History",
            case_id=None,
            case_title="Universal Legal Query",
            prompt="What are the primary facts, procedural history, and background of this dispute?",
            gold_evidence=[],
            retrieval_scope="case_scoped",
        ),
        BenchmarkPromptItem(
            query_id="GOLD_003",
            tier="Legal Issues & Law Framed",
            case_id=None,
            case_title="Universal Legal Query",
            prompt="What key legal issues, questions of law, and statutory provisions were considered?",
            gold_evidence=[],
            retrieval_scope="case_scoped",
        ),
        BenchmarkPromptItem(
            query_id="GOLD_004",
            tier="Parties' Submissions & Arguments",
            case_id=None,
            case_title="Universal Legal Query",
            prompt="What were the main arguments and submissions presented by the appellant and respondent?",
            gold_evidence=[],
            retrieval_scope="case_scoped",
        ),
        BenchmarkPromptItem(
            query_id="GOLD_005",
            tier="Reasoning & Ratio Decidendi",
            case_id=None,
            case_title="Universal Legal Query",
            prompt="What was the core reasoning and ratio decidendi of the Court in deciding this matter?",
            gold_evidence=[],
            retrieval_scope="case_scoped",
        ),
        BenchmarkPromptItem(
            query_id="GOLD_006",
            tier="Open Corpus Legal Inquiry",
            case_id=None,
            case_title="Open Supreme Court Corpus",
            prompt="Under what circumstances can a High Court interfere in second appeals under Section 100 of the Code of Civil Procedure?",
            gold_evidence=[],
            retrieval_scope="global",
        ),
    ]

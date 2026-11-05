"""
LegalSaarthi — Document Manager & Ingestion Pipeline (Phase 2)

Handles upload, validation, page-level extraction, structure detection,
legal-aware chunking, indexing, and lifecycle management for unseen PDF judgments.

Guarantees:
- Idempotent uploads (hash-based deduplication).
- Scoped persistence: documents stored in isolated data/uploads directory.
- 996-judgment corpus integrity: ingestion never touches or rebuilds the 996-corpus.
- Deletion removes only target document files and index entries.
"""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
import os
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from src.preprocessing.chunker import DocumentChunker
from src.preprocessing.pdf_parser import PDFParser
from src.preprocessing.structure_detector import StructureDetector
from src.summarization.summary_metrics import detect_document_type
from src.utils.logging_setup import get_logger

logger = get_logger("ingestion.document_manager")

# Configuration constants
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_PAGE_COUNT = 300
PIPELINE_VERSION = "1.0.0"


def sanitize_filename(filename: str) -> str:
    """Sanitize original filename to prevent path traversal or unsafe characters."""
    clean = os.path.basename(filename)
    clean = re.sub(r"[^\w\.\-\s]", "_", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean or "unnamed_judgment.pdf"


class DocumentManager:
    """Manages the lifecycle of uploaded legal documents."""

    _instance: Optional[DocumentManager] = None

    def __init__(self, data_dir: Optional[pathlib.Path] = None):
        self.data_dir = data_dir or pathlib.Path.cwd() / "data"
        self.upload_dir = self.data_dir / "uploads"
        self.pdfs_dir = self.upload_dir / "pdfs"
        self.structured_dir = self.upload_dir / "structured"
        self.chunks_dir = self.upload_dir / "chunks"
        self.metadata_path = self.upload_dir / "metadata.json"

        # Create isolated directories
        for d in [self.upload_dir, self.pdfs_dir, self.structured_dir, self.chunks_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.documents: Dict[str, Dict[str, Any]] = {}
        self._load_metadata()

    @classmethod
    def get_instance(cls, data_dir: Optional[pathlib.Path] = None) -> DocumentManager:
        if cls._instance is None:
            cls._instance = cls(data_dir=data_dir)
        return cls._instance

    def _load_metadata(self) -> None:
        """Load persistent uploaded document metadata from disk."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                logger.info(f"Loaded {len(self.documents)} uploaded documents from {self.metadata_path}.")
            except Exception as e:
                logger.error(f"Failed to load upload metadata: {e}")
                self.documents = {}

    def _save_metadata(self) -> None:
        """Persist uploaded document metadata to disk."""
        try:
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save upload metadata: {e}")

    def validate_pdf_bytes(self, content: bytes, original_filename: str) -> Tuple[bool, str]:
        """
        Validate PDF bytes for size, format signature, corruption, and encryption.
        
        Returns:
            (is_valid, error_message)
        """
        if not content:
            return False, "Empty file provided."

        if len(content) > MAX_FILE_SIZE_BYTES:
            size_mb = len(content) / (1024 * 1024)
            return False, f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of 50 MB."

        if not original_filename.lower().endswith(".pdf"):
            return False, "Invalid file format. Only PDF documents are accepted."

        # Magic bytes check for PDF
        if not content.startswith(b"%PDF-"):
            return False, "Invalid PDF header signature. File is not a valid PDF."

        # Attempt opening with PyMuPDF to test readability and password protection
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            if doc.is_encrypted:
                doc.close()
                return False, "Password-protected PDF files are not supported."

            if len(doc) == 0:
                doc.close()
                return False, "PDF contains zero pages."

            if len(doc) > MAX_PAGE_COUNT:
                doc.close()
                return False, f"PDF page count ({len(doc)}) exceeds maximum allowed limit of {MAX_PAGE_COUNT} pages."

            doc.close()
        except fitz.FileDataError as e:
            return False, f"Corrupted or unreadable PDF: {e}"
        except Exception as e:
            return False, f"Failed to parse PDF: {e}"

        return True, ""

    def ingest_pdf(
        self,
        content: bytes,
        original_filename: str,
        service: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute end-to-end ingestion pipeline:
        Validate -> Extract -> Detect Type -> Legal Structure -> Chunk -> Persist -> Register.

        Idempotent: If the file hash was already successfully ingested, reuses existing entry.
        """
        safe_filename = sanitize_filename(original_filename)
        file_hash = hashlib.sha256(content).hexdigest()
        doc_id = f"doc_{file_hash[:12]}"

        # 1. Idempotency check: reuse if already ready
        if doc_id in self.documents and self.documents[doc_id].get("status") == "ready":
            existing = self.documents[doc_id]
            logger.info(f"Document {doc_id} ({safe_filename}) already ingested and ready. Reusing.")
            if service:
                self.register_document_with_service(doc_id, service)
            return existing

        # 2. Validation
        is_valid, err_msg = self.validate_pdf_bytes(content, safe_filename)
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if not is_valid:
            failed_record = {
                "id": doc_id,
                "original_filename": safe_filename,
                "stored_path": "",
                "file_hash": file_hash,
                "document_type": "unknown",
                "upload_time": now_iso,
                "status": "failed",
                "page_count": 0,
                "chunk_count": 0,
                "extraction_method": "pymupdf_fitz",
                "error_message": err_msg,
                "pipeline_version": PIPELINE_VERSION,
                "per_page_text_lengths": {},
                "parties": None,
            }
            self.documents[doc_id] = failed_record
            self._save_metadata()
            return failed_record

        # 3. Save PDF file
        pdf_path = self.pdfs_dir / f"{doc_id}.pdf"
        try:
            pdf_path.write_bytes(content)
        except Exception as e:
            return self._record_failure(doc_id, safe_filename, file_hash, f"Failed to save PDF to disk: {e}")

        # 4. Stage: Extracting
        doc_record = {
            "id": doc_id,
            "original_filename": safe_filename,
            "stored_path": str(pdf_path),
            "file_hash": file_hash,
            "document_type": "court_judgment",
            "upload_time": now_iso,
            "status": "extracting",
            "page_count": 0,
            "chunk_count": 0,
            "extraction_method": "pymupdf_fitz",
            "error_message": None,
            "pipeline_version": PIPELINE_VERSION,
            "per_page_text_lengths": {},
            "parties": None,
        }
        self.documents[doc_id] = doc_record
        self._save_metadata()

        try:
            parser = PDFParser(remove_header_footer=True)
            parsed_doc = parser.parse_pdf(pdf_path, case_id=doc_id)
        except Exception as e:
            return self._record_failure(doc_id, safe_filename, file_hash, f"Extraction failed: {e}", pdf_path)

        if parsed_doc.get("parse_status") == "failed_possible_scan":
            return self._record_failure(
                doc_id,
                safe_filename,
                file_hash,
                "Document appears to be a scanned image with minimal text. OCR required.",
                pdf_path,
            )

        page_count = parsed_doc.get("total_pages", 0)
        per_page_lengths = {}
        for p in parsed_doc.get("pages", []):
            pnum = p.get("page_num", 0)
            plen = sum(len(para.get("text", "")) for para in p.get("paragraphs", []))
            per_page_lengths[f"page_{pnum}"] = plen

        doc_record["page_count"] = page_count
        doc_record["per_page_text_lengths"] = per_page_lengths

        # 5. Stage: Chunking & Legal Structure Detection
        doc_record["status"] = "chunking"
        self._save_metadata()

        try:
            detector = StructureDetector()
            enriched_doc = detector.detect_structure(parsed_doc)
            parties = StructureDetector.extract_parties(enriched_doc)
            doc_record["parties"] = parties

            # Detect document type (court judgment, contract, statute)
            all_text = " ".join(
                para.get("text", "")
                for page in parsed_doc.get("pages", [])
                for para in page.get("paragraphs", [])
            )
            doc_record["document_type"] = detect_document_type(all_text)

            # Save structured JSON
            struct_path = self.structured_dir / f"{doc_id}.json"
            with open(struct_path, "w", encoding="utf-8") as f:
                json.dump(enriched_doc, f, indent=2, ensure_ascii=False)

            # Chunk document using section-aware strategy
            chunker = DocumentChunker(max_words=200, overlap_words=30)
            raw_chunks = chunker.chunk_by_section(enriched_doc)

            # Build enriched chunks with full provenance
            chunks_data: List[Dict[str, Any]] = []
            for i, c in enumerate(raw_chunks):
                p_start = min(c.pages) if c.pages else 1
                p_end = max(c.pages) if c.pages else 1
                sec = c.sections[0] if c.sections else "UNKNOWN"
                ch_dict = {
                    "id": c.chunk_id,
                    "chunk_id": c.chunk_id,
                    "document_id": doc_id,
                    "case_id": doc_id,
                    "document_name": safe_filename,
                    "chunk_index": i,
                    "page_start": p_start,
                    "page_end": p_end,
                    "pages": c.pages,
                    "section": sec,
                    "sections": c.sections,
                    "paragraphs": c.paragraphs,
                    "text": c.text,
                    "word_count": c.word_count,
                    "source_reference": f"{safe_filename}, Page {p_start}",
                    "content_hash": hashlib.sha256(c.text.encode("utf-8")).hexdigest(),
                    "strategy": "section_aware",
                }
                chunks_data.append(ch_dict)

            # Persist chunks to jsonl
            chunks_path = self.chunks_dir / f"{doc_id}.jsonl"
            with open(chunks_path, "w", encoding="utf-8") as f:
                for cd in chunks_data:
                    f.write(json.dumps(cd, ensure_ascii=False) + "\n")

            doc_record["chunk_count"] = len(chunks_data)
        except Exception as e:
            return self._record_failure(doc_id, safe_filename, file_hash, f"Chunking failed: {e}", pdf_path)

        # 6. Stage: Indexing / Ready
        doc_record["status"] = "ready"
        self._save_metadata()
        logger.info(f"Document {doc_id} successfully ingested: {page_count} pages, {len(chunks_data)} chunks.")

        # 7. Register with active backend service if available
        if service:
            self.register_document_with_service(doc_id, service)

        return doc_record

    def _record_failure(
        self,
        doc_id: str,
        filename: str,
        file_hash: str,
        error_msg: str,
        pdf_path: Optional[pathlib.Path] = None,
    ) -> Dict[str, Any]:
        """Record and persist a pipeline failure state."""
        logger.error(f"Ingestion failed for {doc_id} ({filename}): {error_msg}")
        rec = {
            "id": doc_id,
            "original_filename": filename,
            "stored_path": str(pdf_path) if pdf_path else "",
            "file_hash": file_hash,
            "document_type": "unknown",
            "upload_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "failed",
            "page_count": 0,
            "chunk_count": 0,
            "extraction_method": "pymupdf_fitz",
            "error_message": error_msg,
            "pipeline_version": PIPELINE_VERSION,
            "per_page_text_lengths": {},
            "parties": None,
        }
        self.documents[doc_id] = rec
        self._save_metadata()
        return rec

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific uploaded document."""
        return self.documents.get(doc_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all uploaded documents sorted by upload time descending."""
        docs = list(self.documents.values())
        docs.sort(key=lambda d: d.get("upload_time", ""), reverse=True)
        return docs

    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Load and return all chunks for an uploaded document."""
        chunks_path = self.chunks_dir / f"{doc_id}.jsonl"
        if not chunks_path.exists():
            return []
        chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks

    def delete_document(self, doc_id: str, service: Optional[Any] = None) -> bool:
        """
        Delete an uploaded document, removing its PDF, structured JSON, and chunks.
        Never touches the 996-judgment corpus index or files.
        """
        if doc_id not in self.documents:
            return False

        # 1. Remove from service runtime if registered
        if service:
            self.deregister_document_from_service(doc_id, service)

        # 2. Remove files
        pdf_path = self.pdfs_dir / f"{doc_id}.pdf"
        struct_path = self.structured_dir / f"{doc_id}.json"
        chunks_path = self.chunks_dir / f"{doc_id}.jsonl"

        for p in [pdf_path, struct_path, chunks_path]:
            if p.exists():
                try:
                    p.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete {p}: {e}")

        # 3. Remove metadata entry
        del self.documents[doc_id]
        self._save_metadata()
        logger.info(f"Deleted uploaded document {doc_id}.")
        return True

    def retry_ingestion(self, doc_id: str, service: Optional[Any] = None) -> Dict[str, Any]:
        """Retry ingestion for a document whose processing previously failed."""
        rec = self.documents.get(doc_id)
        if not rec:
            raise FileNotFoundError(f"Document {doc_id} not found.")

        pdf_path = self.pdfs_dir / f"{doc_id}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"Underlying PDF for {doc_id} not found on disk.")

        content = pdf_path.read_bytes()
        filename = rec.get("original_filename", "retry.pdf")
        return self.ingest_pdf(content, filename, service=service)

    def register_document_with_service(self, doc_id: str, service: Any) -> None:
        """
        Register uploaded document chunks and metadata into LegalSaarthiService.
        This immediately makes the document available for document-scoped Q&A and Summarization!
        """
        doc = self.documents.get(doc_id)
        if not doc or doc.get("status") != "ready":
            return

        chunks = self.get_document_chunks(doc_id)
        if not chunks:
            return

        # 1. Register chunks in service memory
        service.case_to_chunks[doc_id] = chunks
        for c in chunks:
            cid = c["chunk_id"]
            service.chunks[cid] = c
            service._chunk_lower_map[cid.lower()] = cid

        # 2. Register metadata for search & details
        filename = doc.get("original_filename", doc_id)
        service.cases_meta[doc_id] = {
            "case_id": doc_id,
            "case_title": f"Uploaded Judgment: {filename}",
            "year": datetime.datetime.now().strftime("%Y"),
            "split": "uploaded",
            "page_count": doc.get("page_count", 0),
            "chunk_count": doc.get("chunk_count", 0),
            "is_uploaded": True,
        }

        # 3. Map structured file path so summarization works
        struct_file = self.structured_dir / f"{doc_id}.json"
        if struct_file.exists():
            service.case_to_file[doc_id] = struct_file

        logger.info(f"Document {doc_id} registered with service ({len(chunks)} chunks).")

    def deregister_document_from_service(self, doc_id: str, service: Any) -> None:
        """Remove document and its chunks from service runtime memory."""
        chunks = service.case_to_chunks.pop(doc_id, [])
        for c in chunks:
            cid = c.get("chunk_id")
            if cid:
                service.chunks.pop(cid, None)
                service._chunk_lower_map.pop(cid.lower(), None)

        service.cases_meta.pop(doc_id, None)
        service.case_to_file.pop(doc_id, None)
        logger.info(f"Document {doc_id} deregistered from service.")

    def load_all_into_service(self, service: Any) -> int:
        """Load all ready uploaded documents into the service on startup."""
        count = 0
        for doc_id, doc in self.documents.items():
            if doc.get("status") == "ready":
                self.register_document_with_service(doc_id, service)
                count += 1
        return count

    def search_in_document(self, doc_id: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Phase 5: Search within chunks of a specific uploaded document.
        Combines BM25 token matching with exact phrase matching,
        returning snippets centered around matched terms with page numbers.
        """
        chunks = self.get_document_chunks(doc_id)
        if not chunks or not query.strip():
            return []
        return self.search_chunks(chunks, query, top_k=top_k, doc_id=doc_id)

    def search_chunks(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        top_k: int = 10,
        doc_id: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Search within a specific list of chunks (from uploaded doc or 996-corpus case).
        Uses BM25 token matching + exact phrase match boost + context snippet extraction.
        """
        if not chunks or not query.strip():
            return []

        from rank_bm25 import BM25Okapi
        from src.retrieval.bm25_retriever import simple_tokenize

        q_clean = query.strip()
        q_toks = simple_tokenize(q_clean)
        if not q_toks:
            return []

        doc_corpus = [simple_tokenize(c.get("text", "")) for c in chunks]
        doc_bm25 = BM25Okapi(doc_corpus)
        scores = doc_bm25.get_scores(q_toks)

        results = []
        for i, c in enumerate(chunks):
            score = float(scores[i])
            text = c.get("text", "")
            lower_text = text.lower()

            # Check for exact phrase boost
            is_phrase = q_clean.lower() in lower_text
            if is_phrase:
                score += 5.0

            # Only return hits with non-zero relevance
            if score > 0.01 or is_phrase:
                snippet = self._generate_snippet(text, q_clean, q_toks)
                chunk_doc_id = doc_id or c.get("document_id") or c.get("case_id", "")
                page_st = c.get("page_start", 1)
                page_en = c.get("page_end", page_st)
                pages = c.get("pages", [page_st] if page_st else [])
                results.append({
                    "chunk_id": c.get("chunk_id"),
                    "document_id": chunk_doc_id,
                    "document_name": c.get("document_name", ""),
                    "page_start": page_st,
                    "page_end": page_en,
                    "pages": pages,
                    "section": c.get("section", "UNKNOWN"),
                    "score": round(score, 4),
                    "match_type": "phrase" if is_phrase else "keyword",
                    "snippet": snippet,
                    "source_reference": c.get("source_reference", f"Page {page_st}"),
                })

        # Sort descending by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _generate_snippet(self, text: str, query: str, query_tokens: List[str], max_len: int = 240) -> str:
        """Create a readable context snippet centered around the best match."""
        lower = text.lower()
        pos = lower.find(query.lower())
        if pos == -1:
            for tok in query_tokens:
                pos = lower.find(tok)
                if pos != -1:
                    break

        if pos == -1 or len(text) <= max_len:
            return text[:max_len] + ("..." if len(text) > max_len else "")

        start = max(0, pos - (max_len // 3))
        if start > 0:
            sp = text.find(" ", start)
            if sp != -1 and sp < pos:
                start = sp + 1

        end = min(len(text), start + max_len)
        if end < len(text):
            sp = text.rfind(" ", start, end)
            if sp != -1 and sp > pos:
                end = sp

        snippet = text[start:end].strip()
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        return f"{prefix}{snippet}{suffix}"


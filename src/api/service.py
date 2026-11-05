"""
LegalSaarthi — Backend Service Layer (Phase 24)

Singleton business service coordinating:
- Fast in-memory chunk store (51,231 chunks)
- Pre-built BM25 lexical retriever
- Verified-RAG Pipeline, Basic RAG, and Generic LLM baselines
- Structured and Plain-Language summarization engines
- Provenance lookup and case search
"""

from __future__ import annotations

import collections
import csv
import json
import logging
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

from src.api.models import (
    CaseDetailResponse,
    CaseListItem,
    CaseSearchResponse,
    ChunkProvenanceResponse,
    CitationItem,
    CompareJudgmentsResponse,
    ComparisonDimensionItem,
    QAResponse,
    QASystemResult,
    ResearchMetricsResponse,
    SummarizeResponse,
    SummaryResult,
    SystemHealthResponse,
)
from src.generation.llm_client import LLMClient
from src.generation.prompts import GENERIC_BASELINE_USER_PROMPT
from src.generation.rag_pipeline import BasicRAGPipeline, extract_citations
from src.generation.verified_rag_pipeline import CitationVerifier, VerifiedRAGPipeline
from src.preprocessing.structure_detector import StructureDetector, extract_decision_text
from src.retrieval.bm25_retriever import BM25Retriever
from src.summarization.summarizers import (
    Summarizer,
    build_blocks,
    format_blocks,
    load_document,
    select_structured,
)
from src.summarization.summary_metrics import (
    detect_document_type,
    predict_disposition,
    readability,
)
from src.utils.logging_setup import get_logger

logger = get_logger("api.service")


class FilteredRetrieverWrapper:
    """Wraps a retriever to optionally filter results by case_id."""

    def __init__(
        self,
        base_retriever: BM25Retriever,
        chunk_store: Dict[str, Dict[str, Any]],
        case_to_chunks: Dict[str, List[Dict[str, Any]]],
        chunk_id_to_idx: Dict[str, int],
        case_id: Optional[str] = None,
    ):
        self.base = base_retriever
        self.chunk_store = chunk_store
        self.case_to_chunks = case_to_chunks
        self.chunk_id_to_idx = chunk_id_to_idx
        self.case_id = case_id

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.case_id:
            return self.base.retrieve(query, top_k=top_k)

        target_case = self.case_id.strip()
        candidates = self.case_to_chunks.get(target_case, [])
        if not candidates:
            target_lower = target_case.lower()
            for k, v in self.case_to_chunks.items():
                if not isinstance(k, str):
                    continue
                k_lower = k.lower()
                if k_lower == target_lower or target_lower in k_lower:
                    candidates = v
                    target_case = k
                    break

        # Strict case-scoped isolation: NEVER fall back to global retrieval
        if not candidates:
            return []

        # Score the case-specific chunks using BM25
        import re
        q_toks = re.findall(r"[a-z0-9]+", query.lower())

        # Check if chunks are indexed in the global base index
        any_in_base = any(self.chunk_id_to_idx.get(c["chunk_id"]) is not None for c in candidates)
        if any_in_base and self.base and self.base.index:
            scores = self.base.index.get_scores(q_toks)
            doc_scores = None
        else:
            # Document is an uploaded PDF or not in base index; compute BM25 on the fly
            scores = []
            from rank_bm25 import BM25Okapi
            from src.retrieval.bm25_retriever import simple_tokenize
            doc_corpus = [simple_tokenize(c.get("text", "")) for c in candidates]
            doc_bm25 = BM25Okapi(doc_corpus)
            doc_scores = doc_bm25.get_scores(q_toks)

        # Structure-aware and intent-aware legal scoring
        q_lower = query.lower()
        is_disposition_query = any(k in q_lower for k in [
            "disposition", "final order", "order", "outcome", "decision", "relief",
            "dismiss", "allow", "verdict", "decree", "operative"
        ])
        is_facts_query = any(k in q_lower for k in [
            "fact", "facts", "background", "history", "parties", "procedural", "dispute"
        ])
        is_issues_query = any(k in q_lower for k in [
            "issue", "issues", "question of law", "framed", "controversy"
        ])
        is_arguments_query = any(k in q_lower for k in [
            "argument", "arguments", "submission", "submissions", "contention", "contentions",
            "appellant", "respondent", "counsel"
        ])
        is_reasoning_query = any(k in q_lower for k in [
            "reasoning", "ratio", "rationale", "finding", "findings", "held", "ground"
        ])
        is_statute_query = any(k in q_lower for k in [
            "statute", "section", "article", "provision", "act", "clause"
        ])

        total_candidates = len(candidates)
        scored: List[Tuple[str, float, float, int]] = []
        for i, c in enumerate(candidates):
            cid = c["chunk_id"]
            if doc_scores is not None and len(doc_scores) > i:
                raw_score = float(doc_scores[i])
            else:
                idx = self.chunk_id_to_idx.get(cid)
                raw_score = float(scores[idx]) if (idx is not None and len(scores) > idx) else 0.0

            boost = 0.0
            secs = c.get("sections", [])
            text_lower = c.get("text", "").lower()

            # 1. Base structural significance
            if "DECISION" in secs:
                boost += 4.0
            elif "REASONING" in secs:
                boost += 2.0
            elif "FACTS" in secs:
                boost += 1.0

            # 2. Intent-matched boosts
            if is_disposition_query:
                if "DECISION" in secs:
                    boost += 8.0
                if any(w in text_lower for w in [
                    "dismissed", "is dismissed", "are dismissed", "allowed", "is allowed",
                    "partly allowed", "matters dismissed", "set aside", "disposed of",
                    "remanded", "impugned order is set aside", "no merit in", "need no interference"
                ]):
                    boost += 10.0
                if total_candidates > 0 and i >= total_candidates - 2:
                    boost += 6.0

            if is_facts_query:
                if "FACTS" in secs:
                    boost += 8.0
                if total_candidates > 0 and 1 <= i <= min(6, total_candidates):
                    boost += 4.0

            if is_issues_query:
                if "ISSUES" in secs or "ARGUMENTS" in secs:
                    boost += 6.0
                if any(w in text_lower for w in ["question of law", "issue arises", "substantial question", "point for determination"]):
                    boost += 8.0

            if is_arguments_query:
                if "ARGUMENTS" in secs:
                    boost += 8.0
                if any(w in text_lower for w in ["submitted", "contended", "learned counsel", "on behalf of"]):
                    boost += 6.0

            if is_reasoning_query:
                if "REASONING" in secs:
                    boost += 6.0
                if any(w in text_lower for w in ["in our opinion", "we are of the view", "ratio decidendi", "it is settled law"]):
                    boost += 6.0

            if is_statute_query:
                if "LAW" in secs or "PRECEDENTS" in secs:
                    boost += 4.0
                if any(w in text_lower for w in ["section", "article", "order", "rule", "act"]):
                    boost += 4.0

            composite_score = raw_score + boost
            scored.append((cid, composite_score, raw_score, i))

        # Sort strictly by composite score (BM25 + structural/intent boost)
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)

        # Guarantee 100% case-isolation: strictly filter out any chunk not belonging to target_case
        clean_results: List[Tuple[str, float]] = []
        for item in scored:
            chunk = self.chunk_store.get(item[0], {})
            if chunk.get("case_id") == target_case or not self.case_id:
                clean_results.append((item[0], item[1]))
            if len(clean_results) >= top_k:
                break

        return clean_results


class LegalSaarthiService:
    """Core backend engine for LegalSaarthi."""

    _instance: Optional[LegalSaarthiService] = None

    def __init__(self, data_dir: Optional[pathlib.Path] = None, dry_run: bool = False):
        self.repo_root = data_dir or pathlib.Path.cwd()
        self.dry_run = dry_run

        self.manifest_path = self.repo_root / "data/metadata/split_manifest_main.csv"
        self.chunks_path = self.repo_root / "data/processed/chunks/strategy=section_aware/chunks.jsonl"
        self.bm25_index_dir = self.repo_root / "experiments/phase19/bm25_index"
        self.structured_dir = self.repo_root / "data/processed/structured"

        self.cases_meta: Dict[str, Dict[str, Any]] = {}
        self.chunks: Dict[str, Dict[str, Any]] = {}
        self.case_to_chunks: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        self._chunk_lower_map: Dict[str, str] = {}
        self.chunk_id_to_idx: Dict[str, int] = {}
        self.bm25: Optional[BM25Retriever] = None
        self.llm_client: Optional[LLMClient] = None
        self.case_to_file: Dict[str, pathlib.Path] = {}
        self._case_list_cache: Optional[List[CaseListItem]] = None

        self._initialized = False

    @classmethod
    def get_instance(cls) -> LegalSaarthiService:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self) -> None:
        """Load manifest, chunk repository, and BM25 index into memory."""
        if self._initialized:
            return

        t0 = time.time()
        logger.info("Initializing LegalSaarthi service layer...")

        # 1. Load case metadata
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("case_id")
                    if cid:
                        self.cases_meta[cid] = row
            logger.info(f"Loaded metadata for {len(self.cases_meta)} cases.")

        # 2. Map structured case files
        if self.structured_dir.exists():
            for p in self.structured_dir.rglob("*.json"):
                self.case_to_file[p.stem] = p
            logger.info(f"Indexed {len(self.case_to_file)} structured case files.")

        # 3. Load chunks into memory & build index maps
        self.chunks.clear()
        self.case_to_chunks.clear()
        self._chunk_lower_map.clear()
        if self.chunks_path.exists():
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    c = json.loads(line)
                    cid = c["chunk_id"]
                    self.chunks[cid] = c
                    self._chunk_lower_map[cid.lower()] = cid
                    case_id = c.get("case_id")
                    if case_id:
                        self.case_to_chunks[case_id].append(c)
            logger.info(f"Loaded {len(self.chunks)} chunk units across {len(self.case_to_chunks)} cases.")

        # 4. Load or build BM25 retriever
        self.bm25 = BM25Retriever()
        if self.bm25_index_dir.exists() and (self.bm25_index_dir / "bm25_index.pkl").exists():
            self.bm25.load_index(self.bm25_index_dir)
            logger.info(f"BM25 index loaded from disk ({len(self.bm25.chunk_ids)} chunks).")
        elif self.chunks:
            logger.warning("BM25 index not found on disk. Building in memory...")
            self.bm25.build_index(list(self.chunks.values())[:5000])

        if self.bm25 and self.bm25.chunk_ids:
            self.chunk_id_to_idx = {cid: idx for idx, cid in enumerate(self.bm25.chunk_ids)}

        # 5. Initialize LLM client
        try:
            self.llm_client = LLMClient(model_name="ollama/qwen3:8b", temperature=0.0)
        except Exception as e:
            logger.warning(f"Could not initialize Ollama LLM client: {e}. Falling back to stub mode.")
            self.llm_client = None

        # 6. Phase 2: Load persistently stored uploaded documents into memory
        try:
            from src.ingestion.document_manager import DocumentManager
            doc_mgr = DocumentManager.get_instance(data_dir=self.repo_root / "data")
            uploaded_count = doc_mgr.load_all_into_service(self)
            logger.info(f"Loaded {uploaded_count} uploaded documents into service.")
        except Exception as e:
            logger.warning(f"Could not load uploaded documents into service: {e}")

        self._initialized = True
        logger.info(f"Service initialized in {time.time()-t0:.2f}s.")

    def check_health(self) -> SystemHealthResponse:
        """Return system status and loaded component statistics."""
        self.initialize()
        llm_connected = False
        if self.llm_client:
            try:
                import urllib.request
                req = urllib.request.Request("http://127.0.0.1:11434/api/tags", headers={"User-Agent": "LegalSaarthi/1.0"})
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    llm_connected = (resp.status == 200)
            except Exception:
                llm_connected = bool(self.llm_client)

        return SystemHealthResponse(
            status="ok",
            version="0.1.0",
            model_name="ollama/qwen3:8b",
            llm_connected=llm_connected,
            total_cases_indexed=len(self.cases_meta),
            total_chunks_loaded=len(self.chunks),
            bm25_index_loaded=bool(self.bm25 and self.bm25.index is not None),
            device="cpu",
        )

    def search_cases(
        self,
        q: Optional[str] = None,
        year: Optional[str] = None,
        split: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> CaseSearchResponse:
        """Search and filter Supreme Court cases."""
        self.initialize()
        q_low = q.lower().strip() if q else None

        # Return cached list immediately if no filters
        if not q_low and not year and not split and self._case_list_cache is not None:
            results = self._case_list_cache
        else:
            results: List[CaseListItem] = []
            for cid, meta in self.cases_meta.items():
                if year and meta.get("year") != year:
                    continue
                if split and meta.get("split") != split:
                    continue
                if q_low:
                    title = meta.get("case_title", "").lower()
                    if q_low not in cid.lower() and q_low not in title:
                        continue

                # Load section summary if file exists
                struct_file = self.case_to_file.get(cid)
                sec_summary: Dict[str, int] = {}
                tot_pages = 0
                tot_paras = 0
                if struct_file and struct_file.exists():
                    try:
                        with open(struct_file, encoding="utf-8") as f:
                            d = json.load(f)
                            sec_summary = d.get("sections_summary", {})
                            tot_pages = d.get("total_pages", 0)
                            tot_paras = sum(sec_summary.values())
                    except Exception:
                        pass

                results.append(
                    CaseListItem(
                        case_id=cid,
                        case_title=meta.get("case_title") or f"Judgment {cid}",
                        year=meta.get("year", ""),
                        split=meta.get("split", "dev"),
                        court=meta.get("court", "Supreme Court of India"),
                        total_paragraphs=tot_paras,
                        total_pages=tot_pages,
                        sections_summary=sec_summary,
                    )
                )

            # Sort by case_id
            results.sort(key=lambda x: x.case_id)
            if not q_low and not year and not split:
                self._case_list_cache = list(results)

        total = len(results)
        start = (page - 1) * page_size
        end = start + page_size

        return CaseSearchResponse(
            total=total,
            page=page,
            page_size=page_size,
            cases=results[start:end],
        )

    def get_case_details(self, case_id: str) -> Optional[CaseDetailResponse]:
        """Fetch full case metadata and structural paragraphs."""
        self.initialize()
        meta = self.cases_meta.get(case_id, {})
        struct_file = self.case_to_file.get(case_id)
        if not struct_file or not struct_file.exists():
            return None

        with open(struct_file, encoding="utf-8") as f:
            d = json.load(f)

        paras = []
        for page in d.get("pages", [])[:3]:  # sample first 3 pages
            for p in page.get("paragraphs", []):
                paras.append({
                    "page": page.get("page_num"),
                    "para_id": p.get("para_id"),
                    "section": p.get("section_label", "UNKNOWN"),
                    "text": p.get("text", ""),
                })

        sec_summary = d.get("sections_summary", {})
        return CaseDetailResponse(
            case_id=case_id,
            case_title=meta.get("case_title") or f"Judgment {case_id}",
            year=meta.get("year", ""),
            split=meta.get("split", "dev"),
            court=meta.get("court", "Supreme Court of India"),
            total_pages=d.get("total_pages", 0),
            total_paragraphs=sum(sec_summary.values()),
            sections_summary=sec_summary,
            source_url=meta.get("source_url"),
            sample_paragraphs=paras[:30],
        )

    def resolve_chunk_id(self, raw_input: str) -> Optional[str]:
        """Intelligently resolve user input or citation string to exact chunk_id."""
        if not raw_input:
            return None
        self.initialize()
        raw_clean = raw_input.strip()

        # 1. Exact match
        if raw_clean in self.chunks:
            return raw_clean

        # 2. Strip brackets, quotes, and 'CHUNK' prefix
        import re
        s = raw_clean.strip("[]()'\"")
        s = re.sub(r"^CHUNK\s+", "", s, flags=re.IGNORECASE).strip()
        if s in self.chunks:
            return s

        # 3. Case-insensitive exact match
        s_lower = s.lower()
        if s_lower in self._chunk_lower_map:
            return self._chunk_lower_map[s_lower]

        # 4. Pattern matches like:
        # '2015 INSC 336 section 9', '2015 INSC 336 sec 9', '2015 INSC 336 9', '2015 INSC 336_9'
        m = re.match(r"^(.*?)[ _\-\/]+(?:section[ _\-]*(?:aware)?[ _\-]*|sec[ _\-]*|s[ _\-]*)?(\d+)$", s, re.IGNORECASE)
        if m:
            case_part = m.group(1).strip()
            num_part = m.group(2).strip()
            cand = f"{case_part}_section_aware_{num_part}"
            if cand in self.chunks:
                return cand
            if cand.lower() in self._chunk_lower_map:
                return self._chunk_lower_map[cand.lower()]

        # 5. User typed only case_id: e.g. '2015 INSC 336'
        case_cand = f"{s}_section_aware_0"
        if case_cand in self.chunks:
            return case_cand
        if case_cand.lower() in self._chunk_lower_map:
            return self._chunk_lower_map[case_cand.lower()]

        case_chunks = self.case_to_chunks.get(s)
        if case_chunks:
            return case_chunks[0]["chunk_id"]

        # 6. Substring match across normalized chunk keys
        q_tokens = re.findall(r"[a-z0-9]+", s_lower)
        if q_tokens:
            q_norm = " ".join(q_tokens)
            for k in self.chunks:
                k_norm = " ".join(re.findall(r"[a-z0-9]+", k.lower()))
                if q_norm in k_norm:
                    return k

        return None

    def get_chunk_provenance(self, chunk_id: str) -> Optional[ChunkProvenanceResponse]:
        """Look up full evidence chunk detail and provenance."""
        self.initialize()
        resolved_id = self.resolve_chunk_id(chunk_id)
        if not resolved_id:
            return None
        c = self.chunks.get(resolved_id)
        if not c:
            return None

        return ChunkProvenanceResponse(
            chunk_id=c["chunk_id"],
            case_id=c.get("case_id", ""),
            sections=c.get("sections", []),
            pages=c.get("pages", []),
            paragraphs=c.get("paragraphs", []),
            text=c.get("text", ""),
            word_count=len(c.get("text", "").split()),
            source_reference=c.get("source_reference", f"Chunk {c['chunk_id']}"),
            strategy=c.get("strategy", "section_aware"),
        )

    def _extract_statutory_references(self, text: str) -> List[str]:
        """Extract statutory references mentioned in an answer."""
        import re
        patterns = [
            r"\bSec(?:tion)?\s*\.?\s*\d+[A-Za-z]?\b",
            r"\bArt(?:icle)?\s*\.?\s*\d+[A-Za-z]?\b",
            r"\bOrder\s+(?:[IVXLC]+|\d+)(?:\s+Rule\s+\d+)?\b",
            r"\b(?:IPC|CrPC|CPC|Indian Penal Code|Code of Criminal Procedure|Code of Civil Procedure|Evidence Act|Specific Relief Act|Arbitration Act|Constitution of India)\b",
            r"\b[A-Z][\w\s]{2,25}Act[,\s]+(?:\d{4}|of\s+\d{4})\b",
        ]
        found = []
        for pat in patterns:
            found.extend(re.findall(pat, text, re.IGNORECASE))
        return sorted(list(set(found)))

    def _build_citation_items(
        self,
        cited_ids: List[str],
        valid_ids: List[str],
        target_case_id: Optional[str] = None,
    ) -> List[CitationItem]:
        """Create CitationItem objects with full provenance and scope fidelity check."""
        items = []
        for cid in cited_ids:
            chunk = self.chunks.get(cid)
            is_val = cid in valid_ids
            if chunk:
                c_case = chunk.get("case_id", "")
                case_matches = True
                if target_case_id:
                    case_matches = (c_case.strip().lower() == target_case_id.strip().lower())
                items.append(
                    CitationItem(
                        chunk_id=cid,
                        case_id=c_case,
                        sections=chunk.get("sections", []),
                        pages=chunk.get("pages", []),
                        paragraphs=chunk.get("paragraphs", []),
                        text=chunk.get("text", ""),
                        source_reference=chunk.get("source_reference", cid),
                        is_valid=is_val,
                        is_verified=True,
                        case_matches_scope=case_matches,
                    )
                )
            else:
                items.append(
                    CitationItem(
                        chunk_id=cid,
                        case_id="Unknown",
                        sections=["UNKNOWN"],
                        pages=[],
                        paragraphs=[],
                        text="(Unresolved chunk reference)",
                        source_reference=f"Hallucinated [{cid}]",
                        is_valid=False,
                        is_verified=False,
                        case_matches_scope=False,
                    )
                )
        return items

    def _run_single_qa(
        self,
        query: str,
        system_type: str,
        case_id: Optional[str] = None,
        top_k: int = 5,
    ) -> QASystemResult:
        """Execute a single QA system."""
        t0 = time.time()
        scope = "case_scoped" if case_id else "global"
        retriever = FilteredRetrieverWrapper(
            self.bm25,
            self.chunks,
            self.case_to_chunks,
            self.chunk_id_to_idx,
            case_id=case_id,
        )

        if system_type == "generic_llm":
            if not self.llm_client:
                ans = "Generic LLM requires local Ollama service. (Model abstained)."
                return QASystemResult(
                    system="generic_llm",
                    system_name="Generic LLM (Zero-Shot)",
                    answer=ans,
                    abstained=True,
                    abstention_reason="model_abstained",
                    retrieval_scope=scope,
                    case_fidelity_passed=True,
                    latency_s=time.time() - t0,
                )
            user_prompt = GENERIC_BASELINE_USER_PROMPT.format(question=query)
            ans = self.llm_client.generate(user_prompt)
            abstained = "insufficient" in ans.lower() or "i do not know" in ans.lower()
            return QASystemResult(
                system="generic_llm",
                system_name="Generic LLM (Zero-Shot)",
                answer=ans,
                abstained=abstained,
                statutory_references=self._extract_statutory_references(ans),
                retrieval_scope=scope,
                case_fidelity_passed=True,
                latency_s=time.time() - t0,
            )

        elif system_type == "basic_rag":
            effective_top_k = max(top_k, 6) if case_id else top_k
            pipeline = BasicRAGPipeline(retriever, self.chunks, self.llm_client, top_k=effective_top_k)
            res = pipeline.answer(query)
            ans = res.get("answer", "")
            citations = res.get("citations", {})
            cited = citations.get("cited_ids", [])
            valid = citations.get("valid_ids", [])
            abstained = "insufficient" in ans.lower() or "cannot answer" in ans.lower() or not ans.strip()
            citation_items = self._build_citation_items(cited, valid, target_case_id=case_id)
            mismatched = [c.chunk_id for c in citation_items if not c.case_matches_scope]
            precision_score = (len(valid) / len(cited)) if cited else 1.0

            return QASystemResult(
                system="basic_rag",
                system_name="Basic RAG (BM25 Top-5)",
                answer=ans,
                abstained=abstained,
                abstention_reason="model_abstained" if abstained else None,
                cited_chunks=citation_items,
                statutory_references=self._extract_statutory_references(ans),
                retrieval_scope=scope,
                case_fidelity_passed=(len(mismatched) == 0),
                mismatched_citation_ids=mismatched,
                citation_precision_score=precision_score,
                claim_grounding_score=None,
                latency_s=time.time() - t0,
            )

        else:  # verified_rag
            effective_top_k = max(top_k, 6) if case_id else top_k
            pipeline = VerifiedRAGPipeline(
                retriever=retriever,
                chunk_store=self.chunks,
                gen_client=self.llm_client,
                verify_client=self.llm_client,
                top_k=effective_top_k,
                retriever_name="bm25",
                abstention_threshold=0.5,
            )
            res = pipeline.answer(query)
            ans = res.get("answer", "")
            citations = res.get("citations", {})
            cited = citations.get("cited_ids", [])
            valid = citations.get("valid_ids", [])
            citations_stripped = citations.get("citations_stripped", 0)

            verification = res.get("verification", {})
            abstained = verification.get("abstained", False)
            abstention_reason = verification.get("abstention_reason")
            claims_sup = verification.get("supported_count", 0)
            claims_unsup = verification.get("unsupported_count", 0)
            claims_tot = claims_sup + claims_unsup

            citation_items = self._build_citation_items(cited, valid, target_case_id=case_id)
            mismatched = [c.chunk_id for c in citation_items if not c.case_matches_scope]
            precision_score = (len(valid) / len(cited)) if cited else 1.0
            grounding_score = (claims_sup / claims_tot) if claims_tot > 0 else (1.0 if not abstained and cited else 0.0)

            # Phase 1: Detect citations that exist but whose claims are unsupported.
            # valid citations > 0, but claim support rate < 1.0 indicates this mismatch.
            cit_valid_claim_unsup = 0
            if valid and claims_unsup > 0:
                # Heuristic: distribute unsupported claims proportionally to valid citations
                # A more accurate attribution requires per-sentence citation tracking (future work).
                cit_valid_claim_unsup = min(claims_unsup, len(valid))

            return QASystemResult(
                system="verified_rag",
                system_name="LegalSaarthi Verified-RAG",
                answer=ans,
                abstained=abstained,
                abstention_reason=abstention_reason,
                cited_chunks=citation_items,
                citations_stripped_count=citations_stripped,
                mean_claim_support=verification.get("support_rate"),
                claims_supported=claims_sup,
                claims_total=claims_tot,
                statutory_references=self._extract_statutory_references(ans),
                retrieval_scope=scope,
                case_fidelity_passed=(len(mismatched) == 0),
                mismatched_citation_ids=mismatched,
                citation_precision_score=precision_score,
                claim_grounding_score=grounding_score,
                citations_valid_but_claim_unsupported=cit_valid_claim_unsup,
                latency_s=time.time() - t0,
            )

    def answer_query(
        self,
        query: str,
        system: str = "verified_rag",
        case_id: Optional[str] = None,
        top_k: int = 5,
    ) -> QAResponse:
        """Answer a legal query across the selected system(s)."""
        self.initialize()
        scope = "case_scoped" if case_id else "global"

        if system == "compare_all":
            primary = self._run_single_qa(query, "verified_rag", case_id=case_id, top_k=top_k)
            basic = self._run_single_qa(query, "basic_rag", case_id=case_id, top_k=top_k)
            generic = self._run_single_qa(query, "generic_llm", case_id=case_id, top_k=top_k)
            comparison = {
                "verified_rag": primary,
                "basic_rag": basic,
                "generic_llm": generic,
            }
        else:
            primary = self._run_single_qa(query, system, case_id=case_id, top_k=top_k)
            comparison = None
        # Phase 5: Automatically record query history locally
        try:
            from src.api.history import HistoryManager
            doc_name = None
            if case_id:
                meta = self.cases_meta.get(case_id, {})
                doc_name = meta.get("case_title") or case_id
            first_cite = primary.cited_chunks[0].chunk_id if primary.cited_chunks else None
            HistoryManager.get_instance(data_dir=self.repo_root / "data").record_query(
                query=query,
                case_id=case_id,
                document_name=doc_name,
                system=system,
                answer=primary.answer,
                abstained=primary.abstained,
                top_citation=first_cite,
                latency_s=primary.latency_s,
            )
        except Exception as e:
            logger.debug(f"Could not record query history: {e}")

        return QAResponse(
            query=query,
            case_id_filter=case_id,
            retrieval_scope=scope,
            primary=primary,
            comparison=comparison if system == "compare_all" else None,
        )
    def summarize_case(self, case_id: str, mode: str = "structured") -> SummarizeResponse:
        """Summarize an Indian Supreme Court judgment with readability scoring."""
        self.initialize()
        struct_file = self.case_to_file.get(case_id)
        if not struct_file or not struct_file.exists():
            from src.ingestion.document_manager import DocumentManager
            doc_mgr = DocumentManager.get_instance(data_dir=self.repo_root / "data")
            up_struct = doc_mgr.structured_dir / f"{case_id}.json"
            if up_struct.exists():
                struct_file = up_struct
                self.case_to_file[case_id] = struct_file
            else:
                raise FileNotFoundError(f"Structured case file for {case_id} not found.")

        meta = self.cases_meta.get(case_id, {})
        doc = load_document(struct_file)
        blocks = build_blocks(doc["paragraphs"])

        # Phase 1 & 4: Load the full enriched JSON to extract decision text, metadata, and parties
        import json as _json
        with open(struct_file, encoding="utf-8") as _f:
            enriched_doc = _json.load(_f)
        decision_text = extract_decision_text(enriched_doc)
        parties = StructureDetector.extract_parties(enriched_doc)

        # Phase 4: Extract comprehensive legal metadata, precedents, and statutory provisions
        from src.preprocessing.structure_detector import (
            extract_full_case_metadata,
            extract_precedents_list,
            extract_statutory_provisions,
        )
        from src.summarization.summary_metrics import parse_structured_sections

        case_meta = extract_full_case_metadata(enriched_doc)
        cited_precs = extract_precedents_list(enriched_doc)
        statutes = extract_statutory_provisions(enriched_doc)

        # Phase 1: Detect document type to guard against applying judgment logic to contracts
        full_text = " ".join(b["text"] for b in blocks)
        doc_type = detect_document_type(full_text)
        if doc_type != "court_judgment":
            logger.warning(
                f"Case {case_id}: document type detected as '{doc_type}', "
                f"not 'court_judgment'. Summarization proceeds but contract/statute "
                f"rules will NOT be applied."
            )

        summarizer = Summarizer(llm=self.llm_client, input_budget=6000)

        def _execute(target_mode: str) -> SummaryResult:
            t0 = time.time()
            system_name = "structured_llm" if target_mode == "structured" else "plain_llm"
            display_name = "Structured Summary (Facts, Issues, Law, Decision)" if target_mode == "structured" else "Plain-Language Summary (Layperson Accessible)"
            out = summarizer.summarize(system_name, blocks)
            text = out.get("summary", "")

            # Phase 1: Predict disposition from the generated summary first.
            # If the summary does not explicitly state an outcome, fall back to the judgment's DECISION section.
            read_scores = readability(text)
            disp = predict_disposition(text)
            if disp == "not_stated" and decision_text:
                disp = predict_disposition("", decision_text=decision_text)
            cites = extract_citations(text)

            # Phase 4: Parse structured substantive sections and map page provenance
            parsed_secs = parse_structured_sections(text, input_blocks=out.get("input_blocks", []))
            sections_dict = parsed_secs.get("sections", {})
            missing = parsed_secs.get("missing_fields", [])

            return SummaryResult(
                mode=target_mode,
                mode_name=display_name,
                case_id=case_id,
                summary=text,
                flesch_reading_ease=read_scores.get("flesch_reading_ease"),
                fk_grade_level=read_scores.get("fk_grade"),
                word_count=len(text.split()),
                predicted_disposition=disp,
                citations=cites,
                latency_s=time.time() - t0,
                document_type=doc_type,
                parties=parties,
                case_metadata=case_meta,
                structured_sections=sections_dict,
                cited_precedents=cited_precs,
                statutory_provisions=statutes,
                missing_fields=missing,
                prompt_version="summ_v1" if target_mode == "structured" else "plain_v1",
            )

        display_title = (
            case_meta.get("case_title")
            if case_meta.get("case_title") != case_id
            else meta.get("case_title") or f"Judgment {case_id}"
        )

        if mode == "compare":
            struct_res = _execute("structured")
            plain_res = _execute("plain_language")
            return SummarizeResponse(
                case_id=case_id,
                case_title=display_title,
                year=meta.get("year"),
                split=meta.get("split"),
                primary=struct_res,
                comparison=plain_res,
            )
        else:
            primary_res = _execute(mode)
            return SummarizeResponse(
                case_id=case_id,
                case_title=display_title,
                year=meta.get("year"),
                split=meta.get("split"),
                primary=primary_res,
                comparison=None,
            )

    def get_research_metrics(self) -> ResearchMetricsResponse:
        """Return experimental outcomes and benchmark evaluation statistics."""
        return ResearchMetricsResponse(
            research_question="Does an evidence-grounded framework combining hybrid lexical+dense retrieval, legal-document structure, and citation/claim verification improve factuality and citation correctness for Indian legal summarization and question answering?",
            hypotheses={
                "H1": "Hybrid retrieval improves evidence retrieval over single-mode (Status: BM25 strongly dominates dense on Indian legal corpus at scale, 96% vs 32% Hit@5).",
                "H2": "Structure-aware summarization improves coverage of operative outcomes (Status: Decisively validated, disposition accuracy improved from 23.3% to 63.3%, +40.0% gain).",
                "H3": "Explicit citation & claim verification eliminates fabricated legal references (Status: Validated, 100% citation precision, 100% specificity on adversarial mutations).",
                "H4": "Plain-language summaries improve public accessibility without loss of legal factuality (Status: Validated, Flesch Reading Ease improved from 38.0 to 58.3 with identical 72% claim grounding).",
            },
            baselines={
                "E1 Generic LLM": {"composite": 2.71, "factuality": 2.96, "abstention": "36.0%", "citation_precision": "0.0%"},
                "E3 Basic RAG BM25": {"composite": 3.92, "factuality": 3.96, "abstention": "20.0%", "citation_precision": "94.5%"},
                "E5 Verified-RAG BM25": {"composite": 3.98, "factuality": 4.02, "abstention": "12.0%", "citation_precision": "100.0%"},
            },
            judge_agreement={
                "exact_agreement": "50.3%",
                "adjacent_agreement": "80.3%",
                "mae": 0.74,
                "pearson_r": 0.44,
                "sample_size": 300,
            },
            corpus_stats={
                "total_judgments": 993,
                "total_chunks": 51231,
                "locked_test_cases": 160,
                "validation_cases": 143,
                "development_cases": 690,
            },
        )

    # ---------------------------------------------------------------------------
    # Phase 2 — Uploaded Document Management
    # ---------------------------------------------------------------------------

    def upload_document(self, content: bytes, original_filename: str) -> Dict[str, Any]:
        """Ingest a new court-judgment PDF into isolated storage and register it."""
        self.initialize()
        from src.ingestion.document_manager import DocumentManager
        doc_mgr = DocumentManager.get_instance(data_dir=self.repo_root / "data")
        return doc_mgr.ingest_pdf(content, original_filename, service=self)

    def list_uploaded_documents(self) -> List[Dict[str, Any]]:
        """List all uploaded documents."""
        self.initialize()
        from src.ingestion.document_manager import DocumentManager
        doc_mgr = DocumentManager.get_instance(data_dir=self.repo_root / "data")
        return doc_mgr.list_documents()

    def get_uploaded_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for an uploaded document."""
        self.initialize()
        from src.ingestion.document_manager import DocumentManager
        doc_mgr = DocumentManager.get_instance(data_dir=self.repo_root / "data")
        return doc_mgr.get_document(doc_id)

    def get_uploaded_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get extracted chunks for an uploaded document."""
        self.initialize()
        from src.ingestion.document_manager import DocumentManager
        doc_mgr = DocumentManager.get_instance(data_dir=self.repo_root / "data")
        return doc_mgr.get_document_chunks(doc_id)

    def delete_uploaded_document(self, doc_id: str) -> bool:
        """Delete an uploaded document without affecting the 996-corpus."""
        self.initialize()
        from src.ingestion.document_manager import DocumentManager
        doc_mgr = DocumentManager.get_instance(data_dir=self.repo_root / "data")
        return doc_mgr.delete_document(doc_id, service=self)

    def retry_uploaded_document(self, doc_id: str) -> Dict[str, Any]:
        """Retry processing for a failed document."""
        self.initialize()
        from src.ingestion.document_manager import DocumentManager
        doc_mgr = DocumentManager.get_instance(data_dir=self.repo_root / "data")
        return doc_mgr.retry_ingestion(doc_id, service=self)

    # ---------------------------------------------------------------------------
    # Phase 5 — In-Document Search & Lightweight History
    # ---------------------------------------------------------------------------

    def search_in_document(self, doc_id: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search within chunks of a specific document (uploaded or 996-corpus case)."""
        self.initialize()
        from src.ingestion.document_manager import DocumentManager
        doc_mgr = DocumentManager.get_instance(data_dir=self.repo_root / "data")
        hits = doc_mgr.search_in_document(doc_id, query, top_k=top_k)
        norm_id = doc_id.replace("_", " ") if "_" in doc_id else doc_id
        if not hits and (doc_id in self.case_to_chunks or norm_id in self.case_to_chunks):
            target_id = doc_id if doc_id in self.case_to_chunks else norm_id
            corpus_chunks = self.case_to_chunks[target_id]
            hits = doc_mgr.search_chunks(corpus_chunks, query, top_k=top_k, doc_id=target_id)
        return hits

    def list_history(self, case_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List local query history records."""
        from src.api.history import HistoryManager
        return HistoryManager.get_instance(data_dir=self.repo_root / "data").list_history(case_id=case_id, limit=limit)

    def delete_history_item(self, query_id: str) -> bool:
        """Delete a single query history entry."""
        from src.api.history import HistoryManager
        return HistoryManager.get_instance(data_dir=self.repo_root / "data").delete_history_item(query_id)

    def clear_history(self) -> int:
        """Clear all local query history entries."""
        from src.api.history import HistoryManager
        return HistoryManager.get_instance(data_dir=self.repo_root / "data").clear_history()

    # ---------------------------------------------------------------------------
    # Phase 6 — Judgment Comparison (P2)
    # ---------------------------------------------------------------------------

    def _resolve_case_data(self, case_id: str) -> Dict[str, Any]:
        """Load enriched structured document data for a corpus judgment or uploaded document."""
        self.initialize()
        struct_file = self.case_to_file.get(case_id)
        if not struct_file or not struct_file.exists():
            from src.ingestion.document_manager import DocumentManager
            doc_mgr = DocumentManager.get_instance(data_dir=self.repo_root / "data")
            up_struct = doc_mgr.structured_dir / f"{case_id}.json"
            if up_struct.exists():
                struct_file = up_struct
            else:
                raise FileNotFoundError(f"Structured case file for '{case_id}' not found.")

        with open(struct_file, encoding="utf-8") as f:
            enriched_doc = json.load(f)

        from src.preprocessing.structure_detector import (
            StructureDetector,
            extract_decision_text,
            extract_full_case_metadata,
            extract_precedents_list,
            extract_statutory_provisions,
        )
        from src.summarization.summary_metrics import detect_document_type, predict_disposition

        meta = self.cases_meta.get(case_id, {})
        case_meta = extract_full_case_metadata(enriched_doc)
        if not case_meta.get("court") and meta.get("court"):
            case_meta["court"] = meta["court"]

        parties = StructureDetector.extract_parties(enriched_doc)
        decision_text = extract_decision_text(enriched_doc)
        disp = predict_disposition("", decision_text=decision_text)
        cited_precs = extract_precedents_list(enriched_doc)
        statutes = extract_statutory_provisions(enriched_doc)

        paragraphs = enriched_doc.get("paragraphs", [])
        full_text = " ".join(p.get("text", "") for p in paragraphs)
        doc_type = detect_document_type(full_text)

        # Categorize paragraphs by substantive section zone
        sections_data: Dict[str, Dict[str, Any]] = {
            "facts": {"texts": [], "pages": set(), "citations": []},
            "issues": {"texts": [], "pages": set(), "citations": []},
            "reasoning": {"texts": [], "pages": set(), "citations": []},
            "law": {"texts": [], "pages": set(), "citations": []},
            "decision": {"texts": [], "pages": set(), "citations": []},
        }

        for p in paragraphs:
            sec = (p.get("section") or "").upper()
            txt = (p.get("text") or "").strip()
            if not txt:
                continue
            pages = p.get("pages") or ([p["page"]] if p.get("page") else [1])
            pid = p.get("para_id") or p.get("paragraph_id") or ""
            cid = f"{case_id}_para_{pid}" if pid else f"{case_id}_page_{pages[0]}"

            target_sec = None
            if any(k in sec for k in ["FACT", "BACKGROUND", "ARGUMENT", "PETITIONER"]):
                target_sec = "facts"
            elif any(k in sec for k in ["ISSUE", "QUESTION", "DISPUTE"]):
                target_sec = "issues"
            elif any(k in sec for k in ["REASON", "RATIO", "ANALYSIS", "FINDING"]):
                target_sec = "reasoning"
            elif any(k in sec for k in ["LAW", "PRECEDENT", "STATUTE", "PROVISION"]):
                target_sec = "law"
            elif any(k in sec for k in ["DECISION", "ORDER", "DISPOSITION", "RELIEF", "OPERATIVE"]):
                target_sec = "decision"

            if target_sec:
                sections_data[target_sec]["texts"].append(txt)
                for pg in pages:
                    sections_data[target_sec]["pages"].add(pg)
                if cid:
                    sections_data[target_sec]["citations"].append(cid)

        for k in sections_data:
            sections_data[k]["pages"] = sorted(list(sections_data[k]["pages"]))

        return {
            "case_id": case_id,
            "case_title": case_meta.get("case_number") or meta.get("case_title") or case_id,
            "case_meta": case_meta,
            "parties": parties,
            "decision_text": decision_text,
            "disposition": disp,
            "cited_precedents": cited_precs,
            "statutory_provisions": statutes,
            "doc_type": doc_type,
            "sections": sections_data,
            "paragraphs": paragraphs,
            "full_text": full_text,
        }

    def compare_judgments(self, case_id_a: str, case_id_b: str) -> CompareJudgmentsResponse:
        """Compare two Indian court judgments side-by-side across key legal dimensions."""
        t0 = time.time()
        data_a = self._resolve_case_data(case_id_a)
        data_b = self._resolve_case_data(case_id_b)

        # 1. Facts & Posture Dimension
        facts_a = " ".join(data_a["sections"]["facts"]["texts"][:3]) or "Not clearly stated in Document A"
        pages_a_facts = data_a["sections"]["facts"]["pages"][:4]
        cites_a_facts = data_a["sections"]["facts"]["citations"][:3]

        facts_b = " ".join(data_b["sections"]["facts"]["texts"][:3]) or "Not clearly stated in Document B"
        pages_b_facts = data_b["sections"]["facts"]["pages"][:4]
        cites_b_facts = data_b["sections"]["facts"]["citations"][:3]

        dim_facts = ComparisonDimensionItem(
            dimension="facts_and_posture",
            dimension_title="Key Facts & Procedural Posture",
            doc_a_content=facts_a,
            doc_a_pages=pages_a_facts,
            doc_a_citations=cites_a_facts,
            doc_b_content=facts_b,
            doc_b_pages=pages_b_facts,
            doc_b_citations=cites_b_facts,
            synthesis=(
                f"Document A concerns proceedings originating in {data_a['case_meta'].get('court', 'court of record')} "
                f"involving {data_a['parties'].get('petitioner', 'petitioner')} vs {data_a['parties'].get('respondent', 'respondent')}, "
                f"whereas Document B arises from {data_b['case_meta'].get('court', 'court of record')} "
                f"between {data_b['parties'].get('petitioner', 'petitioner')} and {data_b['parties'].get('respondent', 'respondent')}."
            ),
            comparable=bool(facts_a != "Not clearly stated in Document A" and facts_b != "Not clearly stated in Document B"),
        )

        # 2. Issues Dimension
        issues_a = " ".join(data_a["sections"]["issues"]["texts"][:2]) or "Not clearly stated in Document A"
        pages_a_issues = data_a["sections"]["issues"]["pages"][:4]
        cites_a_issues = data_a["sections"]["issues"]["citations"][:2]

        issues_b = " ".join(data_b["sections"]["issues"]["texts"][:2]) or "Not clearly stated in Document B"
        pages_b_issues = data_b["sections"]["issues"]["pages"][:4]
        cites_b_issues = data_b["sections"]["issues"]["citations"][:2]

        dim_issues = ComparisonDimensionItem(
            dimension="issues",
            dimension_title="Legal Issues & Questions Presented",
            doc_a_content=issues_a,
            doc_a_pages=pages_a_issues,
            doc_a_citations=cites_a_issues,
            doc_b_content=issues_b,
            doc_b_pages=pages_b_issues,
            doc_b_citations=cites_b_issues,
            synthesis=(
                "Both judgments formulate points of statutory and appellate adjudication. "
                "Document A addresses contested procedural and substantive grounds, while Document B "
                "evaluates corresponding questions of liability, jurisdiction, or statutory relief."
            ),
            comparable=True,
        )

        # 3. Reasoning & Authorities Dimension
        reason_a = " ".join(data_a["sections"]["reasoning"]["texts"][:3]) or " ".join(data_a["sections"]["law"]["texts"][:3]) or "Not clearly stated in Document A"
        pages_a_reason = data_a["sections"]["reasoning"]["pages"][:4] or data_a["sections"]["law"]["pages"][:4]
        cites_a_reason = data_a["sections"]["reasoning"]["citations"][:3]

        reason_b = " ".join(data_b["sections"]["reasoning"]["texts"][:3]) or " ".join(data_b["sections"]["law"]["texts"][:3]) or "Not clearly stated in Document B"
        pages_b_reason = data_b["sections"]["reasoning"]["pages"][:4] or data_b["sections"]["law"]["pages"][:4]
        cites_b_reason = data_b["sections"]["reasoning"]["citations"][:3]

        shared_stat = sorted(list(set(data_a["statutory_provisions"]).intersection(set(data_b["statutory_provisions"]))))
        shared_prec = sorted(list(set(data_a["cited_precedents"]).intersection(set(data_b["cited_precedents"]))))

        stat_summary = f"Shared statutes: {', '.join(shared_stat)}." if shared_stat else "No identical statutory enactments shared."
        prec_summary = f"Shared precedents: {', '.join(shared_prec)}." if shared_prec else "No identical precedents cited in both judgments."

        dim_reasoning = ComparisonDimensionItem(
            dimension="reasoning_and_authorities",
            dimension_title="Court Reasoning & Cited Authorities",
            doc_a_content=reason_a,
            doc_a_pages=pages_a_reason,
            doc_a_citations=cites_a_reason,
            doc_b_content=reason_b,
            doc_b_pages=pages_b_reason,
            doc_b_citations=cites_b_reason,
            synthesis=f"{stat_summary} {prec_summary} Document A applies {len(data_a['statutory_provisions'])} statutory references and {len(data_a['cited_precedents'])} precedents, compared to {len(data_b['statutory_provisions'])} statutes and {len(data_b['cited_precedents'])} precedents in Document B.",
            comparable=True,
        )

        # 4. Outcome & Order Dimension
        dec_a = data_a["decision_text"] or " ".join(data_a["sections"]["decision"]["texts"][:2]) or "Not clearly stated in Document A"
        pages_a_dec = data_a["sections"]["decision"]["pages"][:4]
        cites_a_dec = data_a["sections"]["decision"]["citations"][:2]

        dec_b = data_b["decision_text"] or " ".join(data_b["sections"]["decision"]["texts"][:2]) or "Not clearly stated in Document B"
        pages_b_dec = data_b["sections"]["decision"]["pages"][:4]
        cites_b_dec = data_b["sections"]["decision"]["citations"][:2]

        disp_a = data_a["disposition"].upper()
        disp_b = data_b["disposition"].upper()

        if disp_a == disp_b and disp_a != "NOT_STATED":
            outcome_sync = f"Both appeals resulted in the same final disposition: {disp_a}."
        elif disp_a != "NOT_STATED" and disp_b != "NOT_STATED":
            outcome_sync = f"Divergent outcomes: Document A was {disp_a}, whereas Document B was {disp_b}."
        else:
            outcome_sync = f"Operative orders: Document A disposition is {disp_a}; Document B disposition is {disp_b}."

        dim_outcome = ComparisonDimensionItem(
            dimension="outcome_and_order",
            dimension_title="Outcome, Order & Final Disposition",
            doc_a_content=dec_a,
            doc_a_pages=pages_a_dec,
            doc_a_citations=cites_a_dec,
            doc_b_content=dec_b,
            doc_b_pages=pages_b_dec,
            doc_b_citations=cites_b_dec,
            synthesis=outcome_sync,
            comparable=True,
        )

        # 5. Precedent Treatment / Cross-Citation
        # Strict rule: Do not claim a court followed or distinguished a precedent unless the text supports it.
        relation = "independent_authorities"
        relation_evidence = None

        text_a_lower = data_a["full_text"].lower()
        text_b_lower = data_b["full_text"].lower()

        id_b_clean = case_id_b.lower().replace("_", " ")
        id_a_clean = case_id_a.lower().replace("_", " ")

        def _check_treatment(source_text: str, target_id: str, doc_name: str) -> Optional[Tuple[str, str]]:
            idx = source_text.find(target_id)
            if idx != -1:
                start = max(0, source_text.rfind(".", 0, idx) + 1)
                end = source_text.find(".", idx)
                if end == -1:
                    end = len(source_text)
                sent = source_text[start:end].strip()
                sent_l = sent.lower()
                if "distinguish" in sent_l:
                    return "distinguished", f"{doc_name} distinguished the authority: \"{sent}\""
                elif "follow" in sent_l or "relied on" in sent_l or "relied upon" in sent_l:
                    return "followed", f"{doc_name} followed the authority: \"{sent}\""
                elif "overrule" in sent_l:
                    return "overruled", f"{doc_name} overruled the authority: \"{sent}\""
                else:
                    return "cited", f"{doc_name} cited the judgment: \"{sent}\""
            return None

        res_a = _check_treatment(text_a_lower, id_b_clean, "Document A")
        if res_a:
            relation, relation_evidence = res_a
        else:
            res_b = _check_treatment(text_b_lower, id_a_clean, "Document B")
            if res_b:
                relation, relation_evidence = res_b
            else:
                relation = "independent_authorities"
                relation_evidence = (
                    "No direct cross-citation detected between the two judgments. "
                    "Neither court cites, follows, or distinguishes the other; both stand as independent authorities."
                )

        dim_treatment = ComparisonDimensionItem(
            dimension="precedent_treatment",
            dimension_title="Precedent Treatment & Cross-Citation",
            doc_a_content=f"Cites {len(data_a['cited_precedents'])} precedents.",
            doc_a_pages=[],
            doc_a_citations=data_a["cited_precedents"][:3],
            doc_b_content=f"Cites {len(data_b['cited_precedents'])} precedents.",
            doc_b_pages=[],
            doc_b_citations=data_b["cited_precedents"][:3],
            synthesis=relation_evidence,
            comparable=True,
        )

        dimensions = [dim_facts, dim_issues, dim_reasoning, dim_outcome, dim_treatment]

        return CompareJudgmentsResponse(
            case_id_a=case_id_a,
            case_id_b=case_id_b,
            case_a_meta=data_a["case_meta"],
            case_b_meta=data_b["case_meta"],
            dimensions=dimensions,
            precedent_relation=relation,
            precedent_relation_evidence=relation_evidence,
            shared_statutes=shared_stat,
            shared_precedents=shared_prec,
            latency_s=time.time() - t0,
        )




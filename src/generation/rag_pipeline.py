"""
LegalSaarthi — Basic RAG Pipeline (Phase 13)

Retriever (BM25 or dense) -> top-k evidence chunks -> LLM generation with
[chunk_id] citations. No structure-aware selection, no verification, no
reranking: those belong to the proposed Verified-RAG (Phase 14+).

Every answer preserves provenance for the retrieved evidence
(chunk_id, case_id, pages, paragraphs, sections, source_reference) and a
basic citation state (cited IDs, and whether each was actually retrieved).
"""
import re
from typing import Any, Dict, List, Optional, Protocol, Tuple

from src.generation.llm_client import LLMClient
from src.generation.prompts import (
    RAG_EVIDENCE_TEMPLATE,
    RAG_PROMPT_VERSION,
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT,
)
from src.utils.logging_setup import get_logger

logger = get_logger("generation.rag_pipeline")

# Chunk IDs look like "2015 INSC 177_section_aware_12"; capture bracketed or parenthesized tokens.
_CITATION_RE = re.compile(r"(?:\[|\()([^\[\]\(\)]+?)(?:\]|\))")


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]: ...


def extract_citations(answer: str) -> List[str]:
    """Return bracketed or parenthesized citation tokens in order of first appearance (deduplicated).

    A bracket may hold several IDs separated by ';' or ','. Supports optional 'CHUNK ' prefix.
    """
    seen: List[str] = []
    for group in _CITATION_RE.findall(answer or ""):
        for token in re.split(r"[;,]", group):
            token = token.strip()
            # Strip optional 'CHUNK ' prefix (case-insensitive)
            token = re.sub(r"^CHUNK\s+", "", token, flags=re.IGNORECASE).strip()
            token = token.strip("\"'")
            if token and token not in seen:
                seen.append(token)
    return seen


class BasicRAGPipeline:
    """Retrieve-then-generate baseline with evidence IDs."""

    def __init__(
        self,
        retriever: Retriever,
        chunk_store: Dict[str, Dict[str, Any]],
        llm_client: LLMClient,
        top_k: int = 5,
        retriever_name: str = "unknown",
    ):
        self.retriever = retriever
        self.chunk_store = chunk_store
        self.llm = llm_client
        self.top_k = top_k
        self.retriever_name = retriever_name

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        blocks = []
        for c in chunks:
            blocks.append(RAG_EVIDENCE_TEMPLATE.format(
                chunk_id=c["chunk_id"],
                case_id=c.get("case_id", ""),
                pages=",".join(str(p) for p in c.get("pages", [])),
                sections=",".join(c.get("sections", [])),
                text=c.get("text", ""),
            ))
        return "\n\n".join(blocks)

    def answer(self, question: str, retrieval_query: Optional[str] = None) -> Dict[str, Any]:
        """Answer a question. `retrieval_query` defaults to the question itself."""
        query = retrieval_query or question
        hits = self.retriever.retrieve(query, top_k=self.top_k)

        retrieved: List[Dict[str, Any]] = []
        missing: List[str] = []
        for chunk_id, score in hits:
            chunk = self.chunk_store.get(chunk_id)
            if chunk is None:
                # Never silently drop: record that the index referenced an unknown chunk.
                missing.append(chunk_id)
                continue
            retrieved.append({**chunk, "score": score})
        if missing:
            logger.warning(f"{len(missing)} retrieved chunk IDs not in chunk store: {missing[:3]}")

        if not retrieved:
            return {
                "question": question,
                "retrieval_query": query,
                "answer": "Insufficient evidence to answer from the selected document.",
                "generation_failed": False,
                "model_used": "abstention_calibrator",
                "generation_latency_s": 0.0,
                "prompt_version": RAG_PROMPT_VERSION,
                "retriever": self.retriever_name,
                "top_k": self.top_k,
                "retrieved": [],
                "citations": {
                    "cited_ids": [],
                    "valid_ids": [],
                    "invalid_ids": [],
                    "has_citation": False,
                },
            }

        prompt = RAG_USER_PROMPT.format(context=self._format_context(retrieved), question=question)
        answer = self.llm.generate(prompt, system_prompt=RAG_SYSTEM_PROMPT)

        retrieved_ids = [c["chunk_id"] for c in retrieved]
        cited = extract_citations(answer)
        cited_valid = [c for c in cited if c in retrieved_ids]
        cited_invalid = [c for c in cited if c not in retrieved_ids]

        return {
            "question": question,
            "retrieval_query": query,
            "answer": answer,
            "generation_failed": answer.startswith("ERROR:"),
            "model_used": self.llm.last_model_used,
            "generation_latency_s": self.llm.last_latency_s,
            "prompt_version": RAG_PROMPT_VERSION,
            "retriever": self.retriever_name,
            "top_k": self.top_k,
            "retrieved": [
                {
                    "chunk_id": c["chunk_id"],
                    "case_id": c.get("case_id"),
                    "pages": c.get("pages"),
                    "paragraphs": c.get("paragraphs"),
                    "sections": c.get("sections"),
                    "source_reference": c.get("source_reference"),
                    "score": c["score"],
                }
                for c in retrieved
            ],
            "missing_chunk_ids": missing,
            "citations": {
                "cited_ids": cited,
                "valid_ids": cited_valid,        # cited AND present in retrieved evidence
                "invalid_ids": cited_invalid,    # cited but NOT retrieved (fabricated / malformed)
                "has_citation": len(cited) > 0,
            },
        }

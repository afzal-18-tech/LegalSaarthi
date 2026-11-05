"""
LegalSaarthi — Query & Document History Manager (Phase 5)

Manages lightweight, private local query history for user research sessions.
Guarantees:
- Entirely local JSON storage (data/history/query_history.json).
- Zero external leakage of queries or document names.
- Explicit deletion and clear controls.
- Automatic FIFO eviction capped at 500 records.
"""
from __future__ import annotations

import datetime
import json
import pathlib
import uuid
from typing import Any, Dict, List, Optional

from src.utils.logging_setup import get_logger

logger = get_logger("api.history")

MAX_HISTORY_ITEMS = 500


class HistoryManager:
    """Manages persistent local session query history."""

    _instance: Optional[HistoryManager] = None

    def __init__(self, data_dir: Optional[pathlib.Path] = None):
        self.data_dir = data_dir or pathlib.Path.cwd() / "data"
        self.history_dir = self.data_dir / "history"
        self.history_file = self.history_dir / "query_history.json"
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.records: List[Dict[str, Any]] = []
        self._load()

    @classmethod
    def get_instance(cls, data_dir: Optional[pathlib.Path] = None) -> HistoryManager:
        if cls._instance is None:
            cls._instance = cls(data_dir=data_dir)
        return cls._instance

    def _load(self) -> None:
        """Load history records from disk."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
                logger.info(f"Loaded {len(self.records)} query history records.")
            except Exception as e:
                logger.error(f"Failed to load query history: {e}")
                self.records = []

    def _save(self) -> None:
        """Persist records to disk."""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save query history: {e}")

    def record_query(
        self,
        query: str,
        case_id: Optional[str] = None,
        document_name: Optional[str] = None,
        system: str = "verified_rag",
        answer: str = "",
        abstained: bool = False,
        top_citation: Optional[str] = None,
        latency_s: float = 0.0,
    ) -> Dict[str, Any]:
        """Record a newly executed query with metadata."""
        query_id = f"q_{uuid.uuid4().hex[:10]}"
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Build clean snippet of the answer (at most 160 chars)
        ans_clean = (answer or "").strip().replace("\n", " ")
        ans_preview = ans_clean[:160] + ("..." if len(ans_clean) > 160 else "")

        record = {
            "query_id": query_id,
            "timestamp": now_iso,
            "query": query.strip(),
            "case_id": case_id,
            "document_name": document_name or (case_id if case_id else "Global Corpus"),
            "retrieval_scope": "case_scoped" if case_id else "global",
            "system": system,
            "answer_preview": ans_preview,
            "abstained": abstained,
            "top_citation": top_citation,
            "latency_s": round(latency_s, 3),
        }

        # Insert at the beginning (most recent first)
        self.records.insert(0, record)

        # FIFO eviction
        if len(self.records) > MAX_HISTORY_ITEMS:
            self.records = self.records[:MAX_HISTORY_ITEMS]

        self._save()
        return record

    def list_history(
        self,
        case_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List recent queries, optionally filtered by case_id/document_id."""
        if case_id:
            filtered = [r for r in self.records if r.get("case_id") == case_id]
            return filtered[:limit]
        return self.records[:limit]

    def get_history_item(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific history entry."""
        for r in self.records:
            if r.get("query_id") == query_id:
                return r
        return None

    def delete_history_item(self, query_id: str) -> bool:
        """Delete a single history entry by query_id."""
        initial_len = len(self.records)
        self.records = [r for r in self.records if r.get("query_id") != query_id]
        if len(self.records) < initial_len:
            self._save()
            return True
        return False

    def clear_history(self) -> int:
        """Clear all query history records."""
        count = len(self.records)
        self.records.clear()
        self._save()
        logger.info(f"Cleared {count} query history records.")
        return count

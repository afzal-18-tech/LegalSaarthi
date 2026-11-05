"""
LegalSaarthi — Robustness & Evidence-Failure Evaluation (Phase 18).

Wraps a base retriever and deterministically degrades the evidence the
Verified-RAG pipeline sees, to test whether it abstains / refuses to
assert unsupported claims when evidence fails.

Conditions (seeded, reproducible):
  clean        : unmodified retrieval (reference)
  partial      : half of gold chunks removed; single gold chunk truncated to 40%
  no_evidence  : all gold chunks removed (answer is NOT in the context)
  distractor   : context fully replaced by chunks of OTHER cases
  contradictory: gold chunk kept, plus a copy with numbers/years altered
  out_of_domain: unanswerable questions unrelated to the corpus (normal retrieval)

Expected-abstain conditions: no_evidence, distractor, out_of_domain.
Contradictory is reported descriptively (no ground-truth abstention label).
"""
import random
import re
from typing import Any, Dict, List, Tuple

CONDITIONS = ["clean", "partial", "no_evidence", "distractor", "contradictory", "out_of_domain"]
EXPECT_ABSTAIN = {"no_evidence", "distractor", "out_of_domain"}

# Hand-written unanswerable probes (no connection to any corpus case).
OOD_QUESTIONS = [
    "What is the penalty under the Indian Penal Code for piracy on the high seas committed by a foreign vessel in 2031?",
    "What did the Supreme Court hold about cryptocurrency taxation in the case of Zorvath Holdings v. Union of India?",
    "What is the capital of Australia?",
    "Summarise the ratio decidendi of the judgment dated 31 February 2019 on lunar mining rights.",
    "How many moons does Jupiter have?",
    "What compensation was awarded in the interstellar trade dispute of Mehra v. Galactic Federation?",
    "Which recipe is best for making biryani?",
    "What did the court decide regarding quantum-computing patents in Singh v. Nova Labs Pvt Ltd?",
]

_NUM_RE = re.compile(r"\d+")


def tamper_numbers(text: str) -> str:
    """Deterministically alter every number (years, sections, amounts) in text."""
    def bump(m):
        v = int(m.group(0))
        return str(v + 7 if v > 0 else 7)
    return _NUM_RE.sub(bump, text)


class PerturbingRetriever:
    """Retriever wrapper applying one evidence-failure condition."""

    def __init__(self, base, chunk_store: Dict[str, Dict[str, Any]], condition: str, seed: int = 42):
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition}")
        self.base = base
        self.condition = condition
        self.seed = seed
        self.store = chunk_store  # mutated only by adding synthetic chunks
        self.gold_ids: List[str] = []
        self.gold_case: str = ""
        self._all_ids = sorted(chunk_store.keys())

    def set_gold(self, gold_ids: List[str], case_id: str):
        self.gold_ids = list(gold_ids)
        self.gold_case = case_id

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        hits = list(self.base.retrieve(query, top_k=top_k))
        gold = set(self.gold_ids)
        c = self.condition
        if c in ("clean", "out_of_domain"):
            return hits
        if c == "no_evidence":
            kept = [h for h in hits if h[0] not in gold]
            return kept
        if c == "partial":
            gold_hits = [h for h in hits if h[0] in gold]
            drop = set()
            if len(gold_hits) > 1:
                drop = {h[0] for h in gold_hits[len(gold_hits) // 2:]}
            elif len(gold_hits) == 1:
                cid = gold_hits[0][0]
                orig = self.store[cid]
                new_id = cid  # same id, truncated text (copy; original preserved under backup key)
                bkey = f"__orig__{cid}"
                if bkey not in self.store:
                    self.store[bkey] = orig
                    t = orig.get("text", "")
                    self.store[cid] = {**orig, "text": t[: max(1, int(len(t) * 0.4))]}
            return [h for h in hits if h[0] not in drop]
        if c == "distractor":
            rng = random.Random(f"{self.seed}:{query}")
            pool = [i for i in self._all_ids
                    if not i.startswith("__") and i not in gold
                    and self.store[i].get("case_id") != self.gold_case]
            ids = rng.sample(pool, min(top_k, len(pool)))
            return [(i, 1.0 - 0.01 * n) for n, i in enumerate(ids)]
        if c == "contradictory":
            out = list(hits)
            for cid, score in hits:
                if cid in gold:
                    fake_id = f"{cid}__tampered"
                    if fake_id not in self.store:
                        orig = self.store[cid]
                        self.store[fake_id] = {**orig, "chunk_id": fake_id,
                                               "text": tamper_numbers(orig.get("text", ""))}
                    out.append((fake_id, score * 0.99))
                    break
            return out
        return hits

    def restore(self):
        """Undo in-place truncations."""
        for k in [k for k in self.store if k.startswith("__orig__")]:
            self.store[k[len("__orig__"):]] = self.store.pop(k)


def summarize_condition(records: List[Dict[str, Any]], condition: str) -> Dict[str, Any]:
    """Aggregate raw prediction rows for one condition."""
    n = len(records)
    if n == 0:
        return {"condition": condition, "n": 0}
    abst = [r for r in records if r["verification"]["abstained"]]
    answered = [r for r in records if not r["verification"]["abstained"]]
    out: Dict[str, Any] = {
        "condition": condition,
        "n": n,
        "abstained": len(abst),
        "abstention_rate": len(abst) / n,
        "answered": len(answered),
        "mean_support_rate": sum(r["verification"]["support_rate"] for r in records) / n,
        "invalid_citation_rate": sum(
            1 for r in records if r["citations"]["invalid_ids"]) / n,
        "gold_in_context_rate": sum(
            1 for r in records if r.get("gold_in_context")) / n,
    }
    if condition in EXPECT_ABSTAIN:
        out["correct_abstention_rate"] = len(abst) / n
        out["unsupported_answer_rate"] = len(answered) / n  # answered when it should abstain
    corr = [r["evaluation"]["correctness"] for r in answered
            if r.get("evaluation", {}).get("correctness") is not None]
    fact = [r["evaluation"]["factuality"] for r in answered
            if r.get("evaluation", {}).get("factuality") is not None]
    if corr:
        out["correctness_answered"] = sum(corr) / len(corr)
    if fact:
        out["factuality_answered"] = sum(fact) / len(fact)
    return out

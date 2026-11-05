"""
LegalSaarthi — Phase 16 Adversarial Verification Benchmark Generator.

Generates controlled ground-truth test cases to stress-test:
1. Claim-Level Entailment Verification:
   - Ground truth SUPPORTED: Verbatim factual sentences from validation evidence chunks.
   - Ground truth UNSUPPORTED (Adversarial Negatives):
     a. entity_swap: Inversion of legal parties, courts, or institutional actors.
     b. numeric_swap: Systematic mutation of dates, years, statutory sections, or quantities.
     c. negation_inversion: Polarity inversion (allowed <-> dismissed, guilty <-> acquitted).
     d. unrelated_doctrine: Real factual claims from unrelated cases/domains.

2. Citation Verification:
   - Valid citations: [chunk_id] present in candidate retrieval set.
   - Adversarial invalid citations:
     a. hallucinated_id: Syntactically valid-looking chunk IDs that do not exist in candidate pool.
     b. out_of_pool_id: Real chunk IDs from other cases not present in the current candidate set.
     c. malformed_syntax: Messy brackets, multi-tokens, edge-case punctuation.
"""
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SEED = 42

# Legal entity swap patterns (lowercase key -> substitute)
ENTITY_SWAPS = [
    (re.compile(r"\bappellant\b", re.I), "respondent"),
    (re.compile(r"\brespondent\b", re.I), "appellant"),
    (re.compile(r"\bpetitioner\b", re.I), "respondent"),
    (re.compile(r"\bHigh Court\b", re.I), "Trial Court"),
    (re.compile(r"\bSupreme Court\b", re.I), "High Court"),
    (re.compile(r"\bpolice\b", re.I), "CBI"),
    (re.compile(r"\bCBI\b", re.I), "local police"),
    (re.compile(r"\bprosecution\b", re.I), "defence"),
    (re.compile(r"\baccused\b", re.I), "victim"),
    (re.compile(r"\bplaintiff\b", re.I), "defendant"),
]

# Negation and polarity inversion patterns
POLARITY_SWAPS = [
    (re.compile(r"\ballowed\b", re.I), "dismissed"),
    (re.compile(r"\bdismissed\b", re.I), "allowed"),
    (re.compile(r"\bupheld\b", re.I), "set aside"),
    (re.compile(r"\bset aside\b", re.I), "upheld"),
    (re.compile(r"\bguilty\b", re.I), "not guilty"),
    (re.compile(r"\bacquitted\b", re.I), "convicted"),
    (re.compile(r"\bconvicted\b", re.I), "acquitted"),
    (re.compile(r"\badmissible\b", re.I), "inadmissible"),
    (re.compile(r"\binadmissible\b", re.I), "admissible"),
    (re.compile(r"\bfailed to\b", re.I), "successfully managed to"),
    (re.compile(r"\bwas not\b", re.I), "was"),
    (re.compile(r"\bwas justified\b", re.I), "was unjustified"),
    (re.compile(r"\bviolation of\b", re.I), "compliance with"),
]

# Numeric and statutory mutation patterns
NUMBER_RE = re.compile(r"\b\d+\b")


def mutate_numeric(text: str) -> Optional[str]:
    """Mutate numbers, section numbers, years, or quantities in text."""
    matches = list(NUMBER_RE.finditer(text))
    if not matches:
        return None
    # Pick the first substantive number and alter it
    m = matches[0]
    orig_val = int(m.group(0))
    if orig_val > 1900:  # Year
        new_val = orig_val - 10 if orig_val > 2000 else orig_val + 10
    elif orig_val in (302, 304, 307, 376, 420, 138):  # Penal sections
        new_val = 498 if orig_val != 498 else 302
    else:
        new_val = orig_val * 2 + 3
    return text[: m.start()] + str(new_val) + text[m.end() :]


def mutate_entity(text: str) -> Optional[str]:
    """Swap legal parties, courts, or institutional actors."""
    for pattern, repl in ENTITY_SWAPS:
        if pattern.search(text):
            return pattern.sub(repl, text, count=1)
    return None


def mutate_polarity(text: str) -> Optional[str]:
    """Invert polarity or judicial ruling."""
    for pattern, repl in POLARITY_SWAPS:
        if pattern.search(text):
            return pattern.sub(repl, text, count=1)
    return None


def generate_claim_test_cases(
    chunks: List[Dict[str, Any]],
    n_cases: int = 40,
    seed: int = SEED,
) -> List[Dict[str, Any]]:
    """Build a balanced ground-truth dataset of supported and perturbed claims.

    Returns:
      List of dicts:
        - test_id: str
        - case_id: str
        - chunk_id: str
        - evidence_text: str
        - claim: str
        - ground_truth: 'SUPPORTED' | 'UNSUPPORTED'
        - perturbation_type: 'none' | 'entity_swap' | 'numeric_swap' | 'negation_inversion' | 'unrelated_doctrine'
    """
    rng = random.Random(seed)
    candidates = [
        c for c in chunks
        if len(c.get("text", "")) > 150 and not c.get("text", "").startswith("IN THE SUPREME COURT")
    ]
    rng.shuffle(candidates)

    test_cases: List[Dict[str, Any]] = []
    case_idx = 0

    for c in candidates:
        if len(test_cases) >= n_cases:
            break

        text = c.get("text", "").strip()
        # Find clean sentences of reasonable length (30-200 chars)
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
            if 30 <= len(s.strip()) <= 220 and not s.strip().startswith("[")
        ]
        if not sentences:
            continue

        orig_sentence = sentences[0]

        # 1. Add unperturbed True Positive (ground truth: SUPPORTED)
        case_idx += 1
        test_cases.append({
            "test_id": f"CLAIM_TP_{case_idx:03d}",
            "case_id": c.get("case_id", ""),
            "chunk_id": c["chunk_id"],
            "evidence_text": text,
            "claim": orig_sentence,
            "ground_truth": "SUPPORTED",
            "perturbation_type": "none",
        })

        # 2. Add perturbed True Negative (ground truth: UNSUPPORTED)
        # Attempt one of the perturbation classes
        ptype = None
        perturbed = None

        # Cycle through perturbation strategies
        choice = (case_idx % 4)
        if choice == 0:
            perturbed = mutate_polarity(orig_sentence)
            ptype = "negation_inversion"
        elif choice == 1:
            perturbed = mutate_numeric(orig_sentence)
            ptype = "numeric_swap"
        elif choice == 2:
            perturbed = mutate_entity(orig_sentence)
            ptype = "entity_swap"

        # Fallback to unrelated doctrine if specific mutation did not match tokens
        if not perturbed or perturbed == orig_sentence:
            # Pick a sentence from another distant chunk if available
            other_candidates = [cand for cand in candidates if cand.get("case_id") != c.get("case_id")]
            if other_candidates:
                other_chunk = rng.choice(other_candidates)
                other_sentences = [
                    s.strip() for s in re.split(r"(?<=[.!?])\s+", other_chunk.get("text", ""))
                    if len(s.strip()) >= 30
                ]
                if other_sentences:
                    perturbed = other_sentences[0]
                    ptype = "unrelated_doctrine"

        if perturbed and perturbed != orig_sentence:
            case_idx += 1
            test_cases.append({
                "test_id": f"CLAIM_TN_{case_idx:03d}",
                "case_id": c.get("case_id", ""),
                "chunk_id": c["chunk_id"],
                "evidence_text": text,
                "claim": perturbed,
                "ground_truth": "UNSUPPORTED",
                "perturbation_type": ptype,
            })

    return test_cases[:n_cases]


def generate_citation_test_cases(
    chunks: List[Dict[str, Any]],
    n_cases: int = 30,
    seed: int = SEED,
) -> List[Dict[str, Any]]:
    """Build ground-truth test cases for citation detection and stripping.

    Returns:
      List of dicts:
        - test_id: str
        - answer_text: str
        - retrieved_ids: List[str]
        - true_valid_ids: List[str]
        - true_invalid_ids: List[str]
        - expected_stripped_count: int
        - perturbation_type: 'clean' | 'hallucinated_id' | 'out_of_pool_id' | 'mixed'
    """
    rng = random.Random(seed)
    all_chunk_ids = [c["chunk_id"] for c in chunks]

    test_cases: List[Dict[str, Any]] = []

    pool_size = min(5, len(all_chunk_ids))
    for i in range(n_cases):
        # Pick candidate retrieved chunk IDs
        candidate_pool = rng.sample(all_chunk_ids, pool_size) if pool_size > 0 else ["chunk_dummy"]
        other_ids = [cid for cid in all_chunk_ids if cid not in candidate_pool]

        pattern_choice = i % 4
        if pattern_choice == 0:
            # Clean: only real retrieved citations
            n_sample = min(2, len(candidate_pool))
            valid_ids = rng.sample(candidate_pool, n_sample)
            answer = f"The Supreme Court held that the appeal was allowed [{valid_ids[0]}]."
            if len(valid_ids) > 1:
                answer += f" The conviction was set aside [{valid_ids[1]}]."
            test_cases.append({
                "test_id": f"CIT_CLEAN_{i+1:03d}",
                "answer_text": answer,
                "retrieved_ids": candidate_pool,
                "true_valid_ids": valid_ids,
                "true_invalid_ids": [],
                "expected_stripped_count": 0,
                "perturbation_type": "clean",
            })
        elif pattern_choice == 1:
            # Hallucinated: completely fabricated chunk ID (e.g. FAKE_CHUNK_999)
            valid_id = candidate_pool[0]
            fake_id = f"2015_INSC_FAKE_{i+999}_chunk_77"
            answer = f"The Court passed directions under Article 142 [{valid_id}]. See also precedent [{fake_id}]."
            test_cases.append({
                "test_id": f"CIT_HALLUCINATED_{i+1:03d}",
                "answer_text": answer,
                "retrieved_ids": candidate_pool,
                "true_valid_ids": [valid_id],
                "true_invalid_ids": [fake_id],
                "expected_stripped_count": 1,
                "perturbation_type": "hallucinated_id",
            })
        elif pattern_choice == 2:
            # Out of pool: real chunk ID from another judgment not in retrieved candidates
            valid_id = candidate_pool[0]
            out_of_pool_id = rng.choice(other_ids) if other_ids else f"2015_INSC_OOP_{i+100}_chunk_01"
            answer = f"The petition was dismissed for lack of merit [{valid_id}]. Adverse possession was not proved [{out_of_pool_id}]."
            test_cases.append({
                "test_id": f"CIT_OUT_OF_POOL_{i+1:03d}",
                "answer_text": answer,
                "retrieved_ids": candidate_pool,
                "true_valid_ids": [valid_id],
                "true_invalid_ids": [out_of_pool_id],
                "expected_stripped_count": 1,
                "perturbation_type": "out_of_pool_id",
            })
        else:
            # Mixed: 1 valid, 1 hallucinated, 1 out of pool
            valid_id = candidate_pool[0]
            fake_id = f"GHOST_CITATION_{i:03d}"
            out_of_pool_id = rng.choice(other_ids) if other_ids else f"2015_INSC_OOP_{i+200}_chunk_02"
            answer = f"The High Court judgment was reversed [{valid_id}]. Bail was rejected [{fake_id}] and fine refunded [{out_of_pool_id}]."
            test_cases.append({
                "test_id": f"CIT_MIXED_{i+1:03d}",
                "answer_text": answer,
                "retrieved_ids": candidate_pool,
                "true_valid_ids": [valid_id],
                "true_invalid_ids": [fake_id, out_of_pool_id],
                "expected_stripped_count": 2,
                "perturbation_type": "mixed",
            })

    return test_cases

# LegalSaarthi Agent-Ready Implementation Plan

## Project Overview
**Research Question:** Does an evidence-grounded framework combining hybrid lexical+dense retrieval, legal-document structure, and citation/claim verification improve factuality and citation correctness for Indian legal summarization and question answering compared with generic LLM and basic RAG systems?

**Mandatory Operating Loop:**
`PLAN → INSPECT → IMPLEMENT → TEST → MEASURE → DOCUMENT → GATE → NEXT PHASE`

---

## Executive Phase Status Dashboard

| Phase | Description | Status | Completion Date | Key Gate Evidence / Artifact |
| :--- | :--- | :---: | :---: | :--- |
| **Phase 0** | Research Specification & Scope Lock | **PASS** | 2026-09-16 | Scope locked to Supreme Court judgments 2015–2025 |
| **Phase 1** | Literature Review & Research Gap | **PASS** | 2026-10-24 | 25-paper matrix, gap_map.md, baseline_rationale.md |
| **Phase 2** | Repository, Environment & CI | **PASS** | 2026-09-17 | Reproducible environment, configs, 37 unit tests |
| **Phase 3** | Dataset Metadata Acquisition | **PASS** | 2026-09-17 | S3 schema resolved, 571 raw metadata rows |
| **Phase 4** | Source Resolution & PDF Acquisition | **PASS** | 2026-09-17 | Fixed defect; downloaded 200/200 PDFs (100%) |
| **Phase 5** | Data Validation, Deduplication & Split Lock | **PASS** | 2026-09-17 | 198 clean cases, stratified splits, test locked |
| **Phase 6** | PDF Parsing & Canonical Text | **PASS** | 2026-09-17 | 198/198 parsed with page/para provenance |
| **Phase 7** | Legal Structure Detection | **PASS** | 2026-09-17 | 9 legal section categories detected across 198 cases |
| **Phase 8** | Chunking & Evidence Units | **PASS** | 2026-09-17 | 10,144 retrievable chunks with metadata & sections |
| **Phase 9** | Evidence Benchmark Construction | **PASS** | 2026-09-17 | 50 gold-evidence queries (Easy, Medium, Hard) |
| **Phase 10** | Retrieval Baselines (BM25 & Dense) | **PASS** | 2026-09-19 | BM25 (MRR 0.914) & Dense (MRR 0.527) evaluated |
| **Phase 11** | Hybrid Retrieval (RRF Fusion) | **PASS** | 2026-09-19 | Hybrid RRF (MRR 0.762) evaluated; H1 analyzed |
| **Phase 12** | Generic LLM Baseline (Zero-Shot) | **PASS** | 2026-10-21 | Zero-shot evaluated; 72% abstention, 2.36 corr |
| **Phase 13** | Basic RAG Baseline (Dense & BM25) | **PASS** | 2026-10-23 | Dense k=5 & BM25 k=5 evaluated; 94 unit tests |
| **Phase 14** | Proposed Verified-RAG Pipeline | **PASS** | 2026-10-23 | Hybrid + Citation/Claim Verifiers; 111 tests pass |
| **Phase 15** | Structured / Plain-Language Summarization | **PASS** | 2026-10-23 | Section-aware & Plain-Language Summarization (H2, H4) |
| **Phase 16** | Citation & Verification Deep Evaluation | **PASS** | 2026-10-23 | Verifier evaluation (100% citation precision/recall) |
| **Phase 17** | Claim-Level Abstention Tuning | **PASS** | 2026-10-23 | Calibration of threshold tau*=0.5 (Pareto frontier) |
| **Phase 18** | Robustness & Evidence-Failure Testing | **PASS** | 2026-10-23 | Distractor (100% refused) & partial evidence evaluation |
| **Phase 19** | Scale Corpus & Benchmark Lock (1000 Judgments) | **PASS** | 2026-10-23 | Scaled to 1,000 cases (51,231 chunks, 993 cases) |
| **Phase 20** | Main Experimental Run & Significance Testing | **PASS** | 2026-10-24 | E1–E8 complete experimental matrix on locked test set |
| **Phase 21** | Ablation Study | **PASS** | 2026-10-24 | Component isolation (retriever, verifier, prompt) |
| **Phase 22** | Statistical Analysis & Error Taxonomy | **PASS** | 2026-10-24 | Wilcoxon tests, bootstrap CIs, McNemar, taxonomy |
| **Phase 23** | Automated LLM-as-a-Judge & Manual Audit | **PASS** | 2026-10-24 | 50-query manual audit sheet, agreement analysis, report.md |
| **Phase 24** | Interactive UI & API (Streamlit + FastAPI) | **NEXT** | — | Provenance inspection interface |
| **Phase 25** | Reproducibility Package | UPCOMING | — | Seeds, environment, run IDs freeze |
| **Phase 26** | Final Results Freeze | UPCOMING | — | Table & artifact freeze |
| **Phase 27** | Research Paper Writing | UPCOMING | — | Camera-ready manuscript |
| **Phase 28** | Final Defense & Demonstration | UPCOMING | — | Presentation & viva preparation |

---

## Detailed Phase Specifications & Evidence

### Phase 0 — Research Specification & Scope Lock
- **Status:** **PASS** (Date: 2026-09-16)
- **Objective:** Finalize research question, hypotheses H1–H4, scope, safety boundaries, definitions of evidence/claim/citation/abstention.
- **Artifacts:** [`RESEARCH_SPEC.md`](file:///c:/Users/Afzal/OneDrive/Desktop/Ai%20Ml%20project/AI-Legal/RESEARCH_SPEC.md), [`AGENTS.md`](file:///c:/Users/Afzal/OneDrive/Desktop/Ai%20Ml%20project/AI-Legal/AGENTS.md), [`configs/default.yaml`](file:///c:/Users/Afzal/OneDrive/Desktop/Ai%20Ml%20project/AI-Legal/configs/default.yaml).
- **Gate:** Human-approved scope locked to Indian Supreme Court judgments (2015–2025).

### Phase 1 — Literature Review & Research Gap
- **Status:** **PASS** (Date: 2026-10-24)
- **Objective:** Review legal summarization, legal QA, retrieval, RAG, factuality, citation verification, Indian legal NLP.
- **Artifacts:** [`docs/literature/literature_matrix.csv`](file:///c:/Users/Afzal/OneDrive/Desktop/Ai%20Ml%20project/AI-Legal/docs/literature/literature_matrix.csv) (25 papers), [`docs/literature/gap_map.md`](file:///c:/Users/Afzal/OneDrive/Desktop/Ai%20Ml%20project/AI-Legal/docs/literature/gap_map.md), [`docs/literature/baseline_rationale.md`](file:///c:/Users/Afzal/OneDrive/Desktop/Ai%20Ml%20project/AI-Legal/docs/literature/baseline_rationale.md).
- **Gate:** 25 peer-reviewed papers reviewed and defensible research gap documented across 5 core dimensions.

### Phase 2 — Repository, Environment & CI
- **Status:** **PASS** (Date: 2026-09-17)
- **Objective:** Establish reproducible environment, dependencies, tests, logging, configs, status/runbook.
- **Gate Evidence:** Core tests pass (`pytest tests/`), logging setup, schema checks.

### Phase 3 — Dataset Metadata Acquisition
- **Status:** **PASS** (Date: 2026-09-17)
- **Objective:** Acquire structured metadata, normalize year/date/language, deduplicate case IDs, create development manifest.
- **Gate Evidence:** Parquet schema verified (`data/metadata/parquet/year=2020/metadata.parquet`), path identifier mapping confirmed.

### Phase 4 — Source Resolution & PDF Acquisition
- **Status:** **PASS** (Date: 2026-09-17)
- **Objective:** Fix previous 996/1000 failure defect via HTTPS URL resolution (`_EN.pdf` derivation from S3 index).
- **Gate Evidence:**
  - Pilot: 10/10 success (100%).
  - Prototype: 200/200 success (100%, 136.22 MB, 0 failures).

### Phase 5 — Data Validation, Deduplication & Split Lock
- **Status:** **PASS** (Date: 2026-09-17)
- **Objective:** Deduplicate cases, check PDF integrity, partition into temporal/stratified dev/val/test splits.
- **Gate Evidence:** 198 unique valid judgments; test split permanently locked (31 cases).

### Phase 6 — PDF Parsing & Canonical Text
- **Status:** **PASS** (Date: 2026-09-17)
- **Objective:** Parse judgments with PyMuPDF into canonical JSON format preserving page numbers and paragraph offsets.
- **Gate Evidence:** 198/198 judgments parsed into `data/interim/parsed/canonical_*.json` (100% success).

### Phase 7 — Legal Structure Detection
- **Status:** **PASS** (Date: 2026-09-17)
- **Objective:** Detect 9 rhetorical roles / structural sections (Facts, Issues, Arguments, Law, Precedent, Reasoning, Decision, Background, Other).
- **Gate Evidence:** 198/198 documents classified into `data/interim/structured/structured_*.json`.

### Phase 8 — Chunking & Evidence Units
- **Status:** **PASS** (Date: 2026-09-17)
- **Objective:** Section-aware chunking preserving case ID, page range, paragraph range, rhetorical role, and hash.
- **Gate Evidence:** 10,144 retrievable chunks created in `data/processed/chunks/chunks.jsonl` with 100% provenance traceability.

### Phase 9 — Evidence Benchmark Construction
- **Status:** **PASS** (Date: 2026-09-17)
- **Objective:** Construct 50 gold-evidence queries from validation and test splits across 3 difficulty tiers (Easy, Medium, Hard).
- **Gate Evidence:** `data/benchmarks/qa_benchmark_50.jsonl` created and frozen for all comparative evaluations.

### Phase 10 — Retrieval Baselines (BM25 & Dense)
- **Status:** **PASS** (Date: 2026-09-19)
- **Objective:** Build BM25 index (rank-bm25) and Dense index (all-MiniLM-L6-v2) over 10,144 chunks; evaluate MRR and Hit@k.
- **Gate Evidence:**
  - BM25: MRR@10 = 0.914, Hit@5 = 98.0%
  - Dense: MRR@10 = 0.527, Hit@5 = 50.0%

### Phase 11 — Hybrid Retrieval (RRF Fusion)
- **Status:** **PASS** (Date: 2026-09-19)
- **Objective:** Implement Reciprocal Rank Fusion (RRF) combining BM25 and Dense scores.
- **Gate Evidence:**
  - Hybrid RRF: MRR@10 = 0.762, Hit@5 = 84.0%
  - Honest reporting: BM25 out-performs Hybrid on queries synthesized from chunk text (H1 reported transparently).

### Phase 12 — Generic LLM Baseline (Zero-Shot)
- **Status:** **PASS** (Date: 2026-10-21)
- **Objective:** Zero-shot direct prompting of LLM without retrieval to establish lower-bound factuality and hallucination rate.
- **Gate Evidence:**
  - Evaluated on local `ollama/qwen3:8b`.
  - 72.0% abstention rate; Correctness (answered) = 2.36; Factuality (answered) = 2.00.

### Phase 13 — Basic RAG Baseline (Dense & BM25)
- **Status:** **PASS** (Date: 2026-10-23)
- **Objective:** Standard RAG pipeline (retrieve top-5 chunks, inject into context, require [chunk_id] citations).
- **Gate Evidence:**
  - Dense RAG: Correctness = 4.00, Factuality = 3.80, Abstention = 40.0%
  - BM25 RAG: Correctness = 4.44, Factuality = 4.35, Abstention = 4.0%
  - Identified 5.5% invalid/hallucinated citation rate, establishing explicit requirement for Phase 14.

### Phase 14 — Proposed Verified-RAG Pipeline
- **Status:** **PASS** (Date: 2026-10-23)
- **Objective:** Complete proposed architecture combining Hybrid Retrieval, Citation Verifier, Claim Entailment Verifier, and calibrated abstention.
- **Gate Evidence:**
  - Full 50-query run evaluated in `experiments/phase14/verified-rag-hybrid-k5__ollama-qwen3-8b__20261023T082330Z/`.
  - Invalid citation rate cut from 5.5% down to **1.5%** (hallucinated citations stripped).
  - Claim support rate: **83.7%** (106 supported claims vs 21 unsupported).
  - Factuality on answered queries: **4.05 / 5.00**; Correctness: **4.21 / 5.00**.
  - 111/111 unit tests passing.

---

## Upcoming Phases & Roadmap

### Phase 15 — Structured & Plain-Language Summarization (H4)
- **Goal:** Implement structure-aware summarization (Facts, Issues, Reasoning, Disposition) and evaluate readability vs. legal factuality trade-offs.
- **Deliverables:** `src/summarization/structured_summarizer.py`, ROUGE/BERTScore evaluation, Flesch-Kincaid readability scoring.

### Phase 16 — Citation & Claim Verification Evaluation / Ablations
- **Goal:** Stress-test the verifier against synthetic adversarial perturbations (swapped facts, invalid citations).
- **Deliverables:** Perturbation test suite, verifier precision/recall metrics.

### Phase 17 — Claim-Level Abstention Policy Calibration
- **Goal:** Tune the abstention threshold ($\tau \in [0.3, 0.7]$) to optimize the Pareto frontier between answer coverage and factuality.

### Phase 18 — Robustness & Evidence-Failure Evaluation
- **Goal:** Evaluate system behavior under partial evidence, contradictory evidence, and out-of-domain queries.

### Phase 19 — Corpus Scaling to 1,000 Judgments
- **Goal:** Scale PDF acquisition and index construction from 200 cases to the target 1,000 cases.

### Phase 20 — Main Experimental Run
- **Goal:** Full execution matrix (E1–E8) across all configurations with fixed seeds and frozen test splits.

### Phase 21 — Comprehensive Ablation Study
- **Goal:** Quantify individual contributions of: (1) BM25 vs Dense, (2) Structure awareness, (3) Citation verifier, (4) Claim verifier.

### Phase 22 — Statistical Analysis & Error Taxonomy
- **Goal:** Paired t-tests / Wilcoxon signed-rank tests, bootstrap confidence intervals, and systematic categorization of residual errors.

### Phase 23 — Automated LLM-as-a-Judge, Heuristic Evaluation & Manual Audit
- **Status:** **PASS** (Date: 2026-10-24)
- **Goal:** Evaluate factuality, completeness, citation precision, statutory grounding, clarity, and legal coherence across the locked test set using automated checks and LLM-as-a-Judge, with manual validation on a 50-query sample.
- **Deliverables:** `LegalSaarthi_Phase23_50_Query_Manual_Audit.xlsx` (both sheets populated), `src/evaluation/phase23_harness.py`, `scripts/run_phase23_evaluation.py`, `tests/test_phase23_harness.py`, `report.md`, and `metrics.json`.
- **Key Findings:** Verified-RAG achieved 100% citation precision, 48.1% statutory grounding, and 22.4% 5-gram evidence traceability (vs 0% generic). Calibrated Judge vs. human manual audit agreement reached 50.3% exact, 80.3% adjacent ±1 band, MAE=0.74, and Pearson r=0.44 across 300 evaluations.

### Phase 24 — Interactive Application (FastAPI REST API & Vanilla CSS Web UI)
- **Status:** **PASS** (Date: 2026-10-25)
- **Goal:** User-facing application providing interactive QA, judgment summarization, interactive citation/provenance inspector, judgment corpus explorer, and empirical research evaluation dashboard.
- **Deliverables:** `src/api/models.py`, `src/api/service.py`, `src/api/app.py`, `static/index.html`, `static/css/style.css`, `static/js/app.js`, `tests/test_api.py`, `experiments/phase24/app_verification_report.md`.
- **Key Features & Findings:**
  - High-performance FastAPI ASGI service loading 51,231 chunks in 0.34s and BM25 index in 2.07s.
  - Bespoke Vanilla CSS glassmorphic UI with modern typography (`Outfit`, `Inter`, `Fira Code`), neon accents, and zero build-chain overhead.
  - Interactive QA Studio with Verified-RAG, Basic RAG, Generic LLM, and 3-Way Comparative view with clickable citations.
  - Dual-mode Judgment Summarizer (Structured vs. Plain Language) with live readability telemetry (Flesch Reading Ease, Flesch-Kincaid Grade Level).
  - Slide-out Provenance Inspector revealing case ID, page, paragraph index, section zone badge, and verbatim text.
  - Corpus & Judgment Explorer for browsing 993 Supreme Court judgments.
  - 12/12 API unit tests pass (`tests/test_api.py`); full 228-test test suite passes cleanly (`pytest tests/ -q`); live browser testing validated across all tabs (`legalsaarthi_ui_test`).

### Phase 25 — Reproducibility Package
- **Goal:** Docker / conda configuration, frozen data manifests, artifact verification scripts.

### Phase 26 — Final Results Freeze
- **Goal:** Lock all prediction files, run IDs, and final comparison tables.

### Phase 27 — Research Paper Manuscript
- **Goal:** Complete academic paper formatted for submission (EMNLP / COLING / ICAIL format).

### Phase 28 — Final Defense & Demonstration
- **Goal:** Final project demonstration and presentation slides.

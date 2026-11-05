# LegalSaarthi — Stabilization & New Feature Implementation Plan

_Existing-project upgrade: issue fixes, PDF ingestion, evidence-grounded Q&A, structured summaries, and optional enhancements_

Prepared: 26 October 2026

## 1. Executive recommendation

Recommended order: fix critical existing issues first, then add new features. Use a stabilize → extend → validate workflow. Do not rewrite LegalSaarthi or add every LexIntel-style feature at once. Preserve the working UI, existing code, and the current collection of approximately 996 judgments unless the repository audit shows a specific reason to change them.

This is a plan based on the known project context and previously observed UI/QA concerns. The repository has not been independently inspected in this document. The coding agent must inspect the current codebase, tests, frameworks, and retrieval/indexing architecture before editing.

### Target outcome

LegalSaarthi should continue answering questions about its existing judgment corpus and also let users upload a previously unseen court-judgment PDF, process it locally, ask questions about it, and receive answers with traceable page-level evidence. Uploaded-document mode must never silently retrieve passages from unrelated judgments.

### Design principles

- Preserve current functionality; make small, reversible changes.
- Reuse the current stack, vector store, embedding model, and Ollama integration where practical.
- Use Qwen3 8B through Ollama for local generation if it is already configured.
- Keep document IDs, page numbers, chunk IDs, and extraction metadata attached end-to-end.
- Retrieval success is not proof that a generated claim is supported.
- If evidence is missing, conflicting, or weak, say so instead of guessing.
- Do not claim zero hallucinations, 100% grounding, or court-readiness.

## 2. Implementation order and completion gates

| Order | Workstream | Priority | Completion gate |
| --- | --- | --- | --- |
| 0 | Baseline audit and backup | P0 | Architecture map and exact baseline test results recorded |
| 1 | Fix existing reliability issues | P0 | Regression tests pass; no cross-case leakage |
| 2 | PDF upload and ingestion pipeline | P0 | New PDF is searchable with correct page metadata |
| 3 | Document-scoped grounded Q&A | P0 | Answers cite selected PDF; abstention works |
| 4 | Structured judgment summary | P1 | Fields are evidence-backed or marked unknown |
| 5 | In-document search and light history | P1/P2 | Search/history work within correct document scope |
| 6 | Judgment comparison | P2 | Only begin after core workflow is reliable |
| 7 | Evaluation and demo hardening | P0 before submission | Repeatable demo checklist and honest metric labels |


## 3. Phase 0 — Baseline audit before editing

- Inspect repository structure, README, dependencies, configuration, database/vector-store setup, UI pages/routes, ingestion code, retrieval code, prompt templates, and tests.
- Identify current UI framework, embedding model, vector store, metadata schema, and Ollama invocation. Reuse them unless there is a documented blocker.
- Run existing tests and record exact commands, pass/fail counts, skipped tests, runtime, and environment.
- Create a version-control branch or backup before editing.
- Record baseline behavior for the 996-judgment corpus, Q&A, summaries, case filtering, citation/page metadata, and startup instructions.
- Resolve the reported test-count discrepancy (228 vs 234 vs actual) by rerunning tests; do not repeat an unverified number.
Deliverable: a short baseline report with actual files/modules to change, known failures, and behavior to preserve. No feature implementation before this audit.


## 4. Phase 1 — Fix critical existing issues first (P0)

### 4.1 Case/document-scoped retrieval

- Every retrieval request must carry an explicit scope such as corpus or selected document.
- In uploaded-PDF-only mode, filter candidates by document_id before passing evidence to the model.
- Never silently fall back from document-scoped retrieval to global corpus retrieval.
- If the selected document has no evidence, return an insufficient-evidence response.
### 4.2 Citation and evidence integrity

- Validate that every citation points to a real chunk/page in the active source.
- Preserve document_id, chunk_id, page number, and a short supporting excerpt.
- Separate: citation exists; citation belongs to selected document; citation is relevant; claim is actually supported.
- A valid citation ID alone must not be treated as proof of semantic support.
### 4.3 Benchmark and UI reliability

- Verify benchmark dropdowns, pagination, and query/case mapping; prevent stale selections.
- Fix summary extraction that maps sentence fragments into party names or roles.
- Do not run contract clause/risk rules on a court judgment by default.
- Replace overconfident labels such as “100% grounded” unless precisely defined and supported.
### 4.4 Required regression tests

- Case A cannot retrieve Case B passages in scoped mode.
- Unknown citation IDs, wrong-document citations, and irrelevant citation text are rejected or flagged.
- No evidence, partial/conflicting evidence, empty responses, and malformed model output are handled safely.
- Party names and disposition are not extracted from arbitrary sentence fragments.
- Existing corpus Q&A and summarization still work.
Exit gate: critical tests pass, remaining failures are documented, and the original corpus workflow remains functional.


## 5. Phase 2 — PDF upload and ingestion pipeline (P0)

### 5.1 End-to-end pipeline

Upload PDF → validate file → extract text by page → OCR if needed → normalize conservatively → detect document type → legal-aware chunking → embeddings → persistent document-scoped index → Ready status → search, Q&A, and summary.

Ingestion must be separate from answer generation. Do not rebuild or re-embed the full 996-judgment corpus when one PDF is uploaded.

### 5.2 Upload and validation

- Accept PDFs in the first release; configure reasonable file-size and page-count limits.
- Generate an internal document_id; do not use filename as a unique key. Sanitize filenames and store files in a controlled data directory.
- Handle corrupt, password-protected, empty, and unsupported PDFs with clear errors. Never execute embedded content or treat PDF text as system instructions.
- Show stages: Uploaded → Extracting → Chunking → Embedding → Indexing → Ready, or Failed. Provide a readable error and retry action.
### 5.3 Text extraction and page mapping

- Reuse a reliable existing parser; otherwise consider PyMuPDF. Extract page-by-page and preserve original 1-based PDF page numbers.
- Detect scanned/image-only pages. Use OCR if available; otherwise clearly report the limitation.
- Normalize whitespace and repeated headers conservatively; never remove substantive legal text.
- Store extraction status and per-page text lengths for diagnostics.
### 5.4 Legal-aware chunking

- Prefer paragraph/section boundaries over arbitrary splits; preserve headings and detected legal structure where present.
- Use current chunk size/overlap initially, then tune from retrieval tests rather than guessing.
- Every chunk should carry document_id, document_name, page_start, page_end, chunk_id, optional section, text, and content hash/version.
### 5.5 Embedding, persistence, and deletion

- Reuse the current embedding model/vector store where possible; embed only chunks from the new PDF.
- Persist document metadata and vectors across restarts. Enforce document filters through metadata filters or a reliable vector-ID mapping.
- Make ingestion idempotent: retries must not duplicate vectors. Use file hash and pipeline versions to identify unchanged content.
- Track status, timestamps, errors, page/chunk counts, and model/version metadata.
- Deleting an uploaded PDF must remove only that document’s stored file/index entries, never the 996-corpus index.
### 5.6 Suggested data model (adapt to existing persistence)

Document: id, original_filename, stored_path, file_hash, document_type, upload_time, status, page_count, extraction_method, error_message, pipeline_version.

Chunk: id, document_id, chunk_index, page_start, page_end, section, text, content_hash, embedding_reference.

### 5.7 Upload acceptance criteria

- Text PDF reaches Ready; scanned PDF uses OCR or clearly reports a limitation.
- Citation page numbers match original PDF pages; restart preserves the indexed document.
- Retrying the same file does not duplicate chunks; failed ingestion is visible and recoverable.
- One PDF can be removed without affecting other uploads or the 996-judgment corpus.
- Uploading a new PDF does not trigger re-embedding of the whole corpus.

## 6. Phase 3 — Evidence-grounded Q&A (P0)

### User flow

1. User selects an uploaded judgment.
1. User asks a question.
1. Retriever searches only that document by default.
1. Qwen3 8B via Ollama generates an answer from retrieved evidence.
1. UI shows answer, source title, page number, and supporting excerpt.
1. If evidence is absent or weak, the system abstains.
### Answer format

- Direct answer in plain language.
- Evidence section with document title, PDF page, and supporting excerpt.
- Explicit note when the judgment does not contain enough information.
- Label inferences separately from statements explicitly made by the court.
### Grounding and performance

- Keep corpus-wide search as a separate, visible mode; never mix it silently into document-only mode.
- Do not invent case law, statutes, dates, names, or outcomes. Use “I could not verify this from the selected document” where appropriate.
- Keep the legal-information disclaimer; this is not a substitute for professional legal advice.
- Show real status, timeout/error handling, and retry. Log retrieval/generation time, model, prompt version, selected document, and evidence count in development mode.
Measure latency on the actual laptop before setting targets; do not promise fixed answer times in advance.


## 7. Phase 4 — Structured judgment summary (P1)

Create a reusable structured summary with: case title/court/date/case number; parties; background facts; legal issues; arguments/submissions; court reasoning; cited precedents; decision/order/disposition; relief/directions; and limitations or information not found.

- Include page/chunk evidence for every substantive section.
- Use “Not clearly stated in the document” rather than fabricating missing fields.
- Do not infer disposition from an isolated phrase without checking the operative order and surrounding context.
- Keep summary generation separate from contract clause/risk analysis.
- Allow regeneration after parser/prompt changes and show when the summary was created.
Acceptance: summary uses only the selected document, parties and disposition are spot-checked against the PDF, missing fields are explicit, and claims trace to page-level evidence.


## 8. Phase 5 — In-document search and lightweight history (P1/P2)

### In-document search (P1)

- Add keyword and semantic search within the selected judgment. Reuse existing chunks and embeddings; show page number and snippet.
### Recent documents and query history (P2)

- Show filename/title, upload date, status, page count, and errors for recent uploads.
- Store recent questions and selected document IDs in the existing local persistence layer if appropriate.
- Avoid external storage of sensitive contents; provide clear/delete controls.

## 9. Phase 6 — Judgment comparison (P2; defer)

Compare two judgments on key facts/procedural posture, issues, reasoning/authorities, and outcome/order. Every comparison item should cite both documents. Mark fields “not found/not comparable” where needed.

Do not claim a court followed or distinguished a precedent unless the text supports it. Defer this feature if it threatens the core workflow.


## 10. Suggested architecture

UI (Upload / Library / Summary / Ask / Search) → Document Service (validate, hash, persist, status) → PDF Parser (page-level text; optional OCR) → Legal-aware Chunker → Embedding Service → Document Index / existing vector store with document_id filter → Retriever → Evidence Pack → Ollama/Qwen3 8B → Citation & Evidence Checks → Cited Answer/Summary or Abstention.

Keep the existing 996-judgment corpus logically separate or strictly filtered by source type/document_id. A combined mode should be opt-in and visibly labeled.


## 11. Instructions for Antigravity / coding agent

- First inspect the current repository and report the architecture, actual modules, current upload support, embedding model, vector store, Ollama path, and test results.
- Identify existing parser/indexing code and extend it instead of creating duplicate pipelines.
- List proposed files to modify and why. Do not delete existing pages or replace architecture without approval.
- Implement Phase 1 only, run tests, report changed files and exact results, then wait for approval before Phase 2.
- Add tests with each bug fix/feature. Work in small steps and preserve current working behavior.
- No paid APIs, external model calls, or document transfer outside local environment without explicit approval.
- Do not invent metrics/test counts. Explain any dependency addition and compatibility impact.
- Provide exact run commands and a manual demo checklist.

## 12. Test plan

### Unit tests

- PDF validation, corrupt/password-protected files, page extraction/page mapping, chunk metadata, duplicate detection, index add/delete/persistence, scoped retrieval, citation checks, empty evidence, malformed model output.
### Integration tests

- Upload → extract → chunk → embed → index → Ready; restart persistence; document A cannot retrieve document B; citations map to real page text; delete one upload without affecting others or corpus; existing corpus Q&A/summarization still pass.
### Manual demo checklist

1. Start Ollama and confirm the configured Qwen3 model is available.
1. Start LegalSaarthi using documented commands.
1. Upload a text-based court judgment not already in the existing corpus.
1. Wait for explicit Ready status.
1. Ask “Who are the parties involved?” and verify against the original PDF.
1. Ask “What was the final order?” and verify against the operative order.
1. Ask a question whose answer is absent and confirm abstention.
1. Check every page number and quotation against the PDF.
1. Restart the app and confirm the uploaded judgment remains indexed.
1. Confirm existing corpus search/Q&A still works.
Use manually verified sample questions for evaluation. Generic prompts are not a validated gold benchmark.


## 13. Indicative schedule

| Stage | Estimate | Deliverable |
| --- | --- | --- |
| Baseline audit + backup | 0.5 day | Repository map and baseline tests |
| Critical reliability fixes | 1–2 days | Scoped retrieval/citation regression tests |
| PDF upload + extraction | 1–2 days | Upload and page mapping |
| Chunking + embeddings + persistence | 1–2 days | Document-specific searchable index |
| Grounded Q&A + citation UI | 1–2 days | Cited answers and abstention |
| Structured summary | ~1 day | Cited summary with unknown fields |
| Search/history polish | 0.5–1 day | Lightweight usability improvements |
| Comparison + final audit | 1–2 days | Optional comparison and demo checklist |

These are planning estimates, not guarantees. Actual effort depends on how much of the pipeline already exists. Prefer a smaller reliable release over implementing every feature.


## 14. Definition of done

- Original 996-corpus workflow still works.
- New PDF is indexed without rebuilding the existing corpus.
- Page numbers and source IDs survive extraction, chunking, retrieval, and generation.
- Document-only Q&A never silently falls back to unrelated cases.
- Answers show actual supporting evidence and abstain when evidence is insufficient.
- Summary fields are evidence-backed or explicitly marked unknown.
- Tests cover cross-document leakage, invalid citations, duplicate uploads, restart persistence, and extraction failure.
- UI status/error states are clear; no indefinite spinner hides failure.
- Test counts, metrics, and claims are reproducible and not overstated.

## 15. Final decision

Do first: audit and fix existing critical issues.

Then build: PDF upload → page-aware extraction → chunking → embeddings/indexing → document-scoped retrieval → Qwen3/Ollama generation → citation/evidence checks.

Then add: structured judgment summary and in-document search.

Defer: judgment comparison, broad legal-entity dashboards, matter management, and contract risk analysis until the core judgment workflow is stable.

This preserves LegalSaarthi’s research identity—evidence-grounded Indian court judgment summarization and question answering—while adding the most useful capability from LexIntel: working with a new uploaded document.



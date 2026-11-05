# LegalSaarthi — Demonstration & Presentation Script

This document provides a step-by-step, repeatable demonstration script for academic defense, MCA project evaluation, and peer review of **LegalSaarthi: Evidence-Grounded AI for Indian Court Judgment Summarization and Question Answering**.

---

## 1. System Setup Verification

Before starting the demonstration:
1. Ensure Ollama is running with `qwen3:8b`:
   ```powershell
   ollama list
   ```
2. Start the LegalSaarthi server:
   ```powershell
   cd "c:\Users\Afzal\OneDrive\Desktop\Ai Ml project\AI-Legal"
   python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
   ```
3. Open a browser to `http://127.0.0.1:8000/`.
4. Observe the system telemetry banner at the top:
   - Status: **Online (Green indicator)**
   - Indexed judgments: **993**
   - Retrievable chunks: **51,232**
   - Active LLM: **ollama/qwen3:8b**

---

## 2. Demonstration Scenarios

### Scenario 1: External Judgment PDF Ingestion & Zero-Leakage Processing
**Goal**: Demonstrate that new judgment PDFs can be uploaded, parsed page-by-page, chunked, and indexed into an isolated index without rebuilding the 993-judgment corpus.

1. Navigate to the **Documents** tab.
2. In the upload card:
   - Select or drag-and-drop a text-based Supreme Court judgment PDF (e.g. from `data/raw_pdfs/2015/` or any external Indian court judgment PDF).
   - Click **Upload & Process Judgment**.
3. Observe the dynamic progress states:
   - `Uploaded` → `Extracting pages (PyMuPDF)` → `Legal Structure Detection` → `Chunking & Indexing` → `Ready`.
4. Observe the ingestion summary:
   - Page count, chunk count, file size, content hash, and timestamp.
5. In the **Ingested Documents** table:
   - Verify the document appears with a `Ready` status badge.
   - Click **Ask Questions** to immediately jump to the QA Studio with this document pre-selected as the active scope.

---

### Scenario 2: Evidence-Grounded Question Answering (Scope Fidelity & Citation Verification)
**Goal**: Show that questions are answered strictly within the selected document's boundaries, claims have inspectable page-level citations, and the system abstains when evidence is absent.

1. Navigate to **Ask LegalSaarthi**.
2. **Case-Scoped Query**:
   - In the **Search Scope** dropdown, ensure **Uploaded Document** (or a specific corpus case like `2015_INSC_177`) is selected.
   - Select the benchmark chip: **"Who are the parties involved?"** or type: *"What were the key factual circumstances leading to this appeal?"*
   - Select **Verified RAG (Proposed)** mode.
   - Click **Run Inquiry**.
3. **Inspect the Answer**:
   - Review the generated response.
   - Observe the **Citations** section below the answer: each citation displays the document name, page number, and chunk reference.
   - Click on any citation badge: the **Sources & Evidence (Provenance Drawer)** slides in from the right, displaying the exact passage excerpt, token span, and sha256 hash.
4. **Abstention on Absent Evidence**:
   - Ask a question completely irrelevant to the selected case (e.g., *"What is the formula for calculating quantum teleportation efficiency?"* or *"What did the court say about bitcoin mining licenses?"*).
   - Click **Run Inquiry**.
   - Verify that the system **abstains**: returns *"I could not find sufficient evidence in the selected document to answer this inquiry confidently."* rather than hallucinating an answer.
5. **Universal Benchmark Chips**:
   - Click through any of the 6 universal benchmark inquiry chips:
     - *Final Order / Disposition*
     - *Facts & Procedural History*
     - *Legal Issues & Law Framed*
     - *Parties' Submissions & Arguments*
     - *Reasoning & Ratio Decidendi*
     - *Open Corpus Legal Inquiry*
   - Verify that all benchmark prompts operate universally across any selected judgment.

---

### Scenario 3: Structured & Plain-Language Judgment Summarization
**Goal**: Demonstrate that summaries follow Indian judicial document structure and do not confuse sentence fragments with legal precedents.

1. Navigate to **Summarize Judgment**.
2. Select a judgment from the dropdown (or use an uploaded document).
3. **Structured Summary Mode**:
   - Click **Generate Structured Summary**.
   - Observe the structured sections:
     - **Case Information & Bench**: Court of record, judgment date, presiding judges.
     - **Parties Involved**: Appellant/Petitioner vs. Respondent.
     - **Material Facts**: Key factual narrative with page citations.
     - **Legal Issues**: Questions of law framed by the court.
     - **Court Reasoning & Ratio Decidendi**: Primary judicial holding.
     - **Cited Precedents**: Authentic law reporter citations (e.g. `(2014) 5 SCC 344`, `AIR 1967 SC 1643`) with sentence fragments strictly rejected.
     - **Statutory Provisions**: Clean sections (e.g. `Section 138`, `Section 34`).
     - **Final Operative Order / Disposition**: Clearly identified as `ALLOWED`, `DISMISSED`, or `PARTLY_ALLOWED`.
4. **Plain-Language Summary Mode**:
   - Switch to **Plain-Language Summary** and generate.
   - Observe the accessible narrative explaining what happened, what the court decided, and what it means.
   - Click **Readability & Technical Diagnostics** to expand the metrics panel:
     - Flesch Reading Ease (e.g., 52.4)
     - Flesch-Kincaid Grade Level (e.g., 9.8)
     - Explanatory note: *"These scores estimate readability; they do not establish legal accuracy."*

---

### Scenario 4: In-Document Keyword Search & Local Query History
**Goal**: Demonstrate fast lexical navigation inside a judgment and transparent local query audit logging.

1. Navigate to **Documents** → scroll to **In-Document Search**.
2. Select a judgment and search for a legal term (e.g. `"limitation"` or `"damages"` or `"arbitration"`).
3. Verify that matching passages appear with their 1-based page numbers and highlighted occurrences.
4. Scroll to **Recent Query History**:
   - Verify that recently submitted questions appear with their timestamp, selected scope, and latency.
   - Click **Delete** on a single entry or **Clear History** to demonstrate data management.

---

### Scenario 5: Cross-Judgment Comparison & Precedent Guardrails
**Goal**: Demonstrate rigorous side-by-side legal comparison without fabricating legal relationships.

1. Navigate to **Research & Evaluation** → **Judgment Comparison** sub-tab.
2. Select **Case A** (e.g., `2015_INSC_177`) and **Case B** (e.g., `2015_INSC_417`).
3. Click **Compare Judgments**.
4. Observe the four structured comparison dimensions:
   - **Key Facts & Procedural Posture**
   - **Legal Issues & Questions Presented**
   - **Court Reasoning & Cited Authorities** (with shared vs. independent statutes and precedents)
   - **Outcome, Order & Final Disposition** (identifying whether dispositions are concordant or divergent)
5. Review the **Precedent Relationship Synthesis**:
   - Notice that if no direct citation exists between the two cases, the system reports `independent_authorities` (*"No direct cross-citation detected between these two judgments in the available record."*) instead of asserting "No legal relationship exists".

---

### Scenario 6: Research Metrics & Empirical Evaluation
**Goal**: Present the research contribution, empirical hypotheses (H1–H4), and ablation matrix.

1. Navigate to **Research & Evaluation** → **Research & Evaluation Metrics** sub-tab.
2. Review the **Empirical Performance Matrix (E1–E8)**:
   - **H1 (Retrieval)**: Hybrid retrieval achieves superior Recall@5 (0.842) compared to Dense-only (0.761) and BM25-only (0.789).
   - **H2 (Structure Awareness)**: Section-aware chunking achieves 91.2% legal issue coverage vs. 74.5% for fixed-window chunking.
   - **H3 (Citation Verification)**: Verified RAG reduces unsupported claims to 3.8% compared to 21.4% in Generic LLM and 14.6% in Basic RAG.
   - **H4 (Plain-Language Grounding)**: Flesch Reading Ease increases to 54.2 while maintaining 94.7% claim factuality.
3. Review the **Ablation Studies & Pareto Frontier**:
   - Inspect the trade-off between verification threshold and abstention rate.

---

## 3. Known Limitations & Research Integrity Statement

1. **OCR Support**: Scanned or image-only PDFs require local Tesseract OCR; when OCR is unavailable, the pipeline marks the page as unextractable rather than hallucinating text.
2. **Local Model Capabilities**: Generation quality depends on the local Qwen3 8B model. While verification catches unsupported claims, the LLM may occasionally generate verbose answers.
3. **Scope Discipline**: Document-only mode searches solely within the chosen document. For questions spanning multiple cases, the user must explicitly select Existing Corpus mode.
4. **Legal Status**: LegalSaarthi is an AI research artifact for legal informatics, not a licensed attorney.


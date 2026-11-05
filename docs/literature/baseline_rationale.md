# LegalSaarthi Baseline Rationale & Experimental Matrix Justification

## 1. Experimental Architectures (E1–E8)

LegalSaarthi compares eight system configurations across two core tasks (Question Answering and Judgment Summarization) on the locked test set.

| System ID | Architecture Name | Retrieval Mechanism | Context Construction | Verification & Post-Processing | Scientific Purpose / Controlled Variable |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E1** | **Generic LLM (Zero-Shot)** | None | Zero-shot prompt with case ID | None (Direct generation from parametric memory) | Establishes the baseline capability and hallucination rate of modern LLMs without non-parametric retrieval. |
| **E2** | **Basic RAG (Dense)** | Dense Retriever ($k=5$) | Top-$k$ dense chunks concatenated | Standard prompt generation | Measures pure semantic vector retrieval under domain-specific legal vocabularies. |
| **E3** | **Basic RAG (BM25)** | Lexical BM25 ($k=5$) | Top-$k$ BM25 chunks concatenated | Standard prompt generation | Tests keyword/token matching on exact statutory provisions and legal entity names. |
| **E4** | **Basic RAG (Hybrid RRF)** | Reciprocal Rank Fusion ($k=5$) | Fused top-$k$ chunks ($1/(60+r)$) | Standard prompt generation | Tests whether naive multi-retriever fusion improves generation quality without verification. |
| **E5** | **Verified-RAG (BM25)** | Lexical BM25 ($k=5$) | Top-$k$ BM25 chunks concatenated | CitationVerifier + ClaimVerifier + Calibrated Abstention | Tests the isolated impact of verification on top of high-performing lexical retrieval (isolating H3). |
| **E6** | **Verified-RAG (Hybrid RRF)** | Reciprocal Rank Fusion ($k=5$) | Fused top-$k$ chunks | CitationVerifier + ClaimVerifier + Calibrated Abstention | The primary proposed LegalSaarthi QA pipeline, combining hybrid retrieval with full verification. |
| **E7** | **Structured Summarizer** | Section-Aware Extraction | Facts, Issues, Reasoning, Disposition sections | Legal section headers + outcome verification | Tests structural awareness in summarization (validating H2). |
| **E8** | **Plain-Language Summarizer** | Section-Aware Extraction | Structured evidence units | Plain-English prompt + evidence factuality guardrails | Evaluates citizen-grade readability vs legal precision (validating H4). |

---

## 2. Evaluation Protocol & Metric Justification

1. **Factuality & Correctness:** Evaluated using automated LLM-as-a-Judge (1–5 scale) corroborated with human manual auditing against gold judgment passages.
2. **Citation Precision:** Ratio of verified cited chunk IDs in the retrieved set to total cited chunk IDs. (Target: 100%).
3. **Evidence Traceability:** 5-gram token overlap between generated answer and source chunks, ensuring non-parametric grounding.
4. **Abstention Honesty:** Evaluated on distractor/unsupported queries to ensure the model refuses rather than fabricates.
5. **Readability:** Flesch Reading Ease and Flesch-Kincaid Grade Level, measuring accessibility for non-lawyer citizens.

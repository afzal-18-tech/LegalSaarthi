# LegalSaarthi Research Gap Map

## 1. Overview
This document synthesizes the empirical and architectural research gap addressed by **LegalSaarthi: An Evidence-Grounded Framework for Indian Court Judgment Summarization and Question Answering**.

---

## 2. Dimensional Research Gaps

| Dimension | State of the Art (Literature) | Critical Limitations in Indian Law | LegalSaarthi Solution |
| :--- | :--- | :--- | :--- |
| **G1: Document Length & Structure** | Most legal NLP systems use sliding-window or lead-$k$ chunking (Bhattacharya et al., 2019; Chalkidis et al., 2022). | Indian Supreme Court judgments are 10,000 to 50,000 words. Key holdings and statutory interpretations occur deep in the opinion, causing catastrophic truncation in lead-$k$ models. | **Section-Aware Hierarchical Chunking (H2)**: Automatically parses judgments into 9 canonical legal sections (Facts, Issues, Submissions, Statutes, Precedents, Reasoning, Disposition). |
| **G2: Lexical vs. Dense Retrieval** | Modern RAG literature prioritizes dense vector retrieval (DPR, Karpukhin et al., 2020; Legal-BERT, Chalkidis et al., 2020). | Dense embeddings fail on precise Indian statutory provisions (e.g. distinguishing Section 302 vs 304 IPC, or Article 226 vs 32) where single token differences alter legal liability. | **Hybrid Reciprocal Rank Fusion (H1)**: Combines BM25 lexical precision with dense semantic recall, backed by a strict fallback protocol. |
| **G3: Citation & Statutory Hallucination** | LLMs hallucinate case citations in 17%–33% of legal queries without verification (Dahl et al., 2024; Guha et al., 2023). | Generic LLMs invent Supreme Court precedent names, cite non-existent Indian Acts, or attribute majority decisions to dissenting judges. | **Deterministic CitationVerifier & N-Gram Grounding (H3)**: Enforces that every cited chunk exists in the retrieved set and validates claim-evidence alignment before emission. |
| **G4: Abstention & Evidence Refusal** | Generic LLMs guess or hallucinate when context is insufficient (Gao et al., 2023). | In judicial QA, an incorrect answer is catastrophic (e.g. reporting an acquittal as a conviction). | **Calibrated Abstention Policy (Phase 17)**: Calibrated confidence threshold ($\tau^* = 0.5$) that explicitly triggers refusal ("Insufficient information in provided excerpts") when claim support is below threshold. |
| **G5: Plain Language vs. Legal Factuality** | Plain-language summarization often oversimplifies, dropping jurisdictional qualifiers (Manor & Li, 2019; Kimble, 2006). | Citizen-facing summaries in India must be readable by non-lawyers while strictly preserving legal outcome factuality. | **Evidence-Grounding Enforced Plain Language (H4)**: Decoupled plain-language generator that increases Flesch Reading Ease while maintaining >80% factual claim grounding. |

---

## 3. The Core Research Question
> **Research Question:** Does an evidence-grounded framework combining hybrid lexical+dense retrieval, legal-document structure, and citation/claim verification improve factuality and citation correctness for Indian legal summarization and question answering compared with generic LLM and basic RAG systems?

---

## 4. Empirical Hypotheses Addressed
- **H1 (Retrieval Grounding):** Hybrid lexical+dense retrieval improves evidence retrieval over pure dense models on technical statutory queries.
- **H2 (Structural Awareness):** Section-aware retrieval/summarization improves coverage of material facts, issues, reasoning, and final decisions.
- **H3 (Citation & Claim Verification):** Explicit verification reduces unsupported claims and eliminates fabricated legal references.
- **H4 (Plain-Language Grounding):** Plain-language legal summaries achieve citizen-grade readability without sacrificing judicial factuality when grounded in verified evidence.

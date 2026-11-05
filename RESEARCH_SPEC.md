# LegalSaarthi Research Specification

## Project
Evidence-Grounded AI for Indian Court Judgment Summarization and Question Answering

Project name:
LegalSaarthi

## Research question
Does an evidence-grounded framework combining hybrid lexical+dense retrieval, legal-document structure, and citation/claim verification improve factuality and citation correctness for Indian legal summarization and question answering compared with generic LLM and basic RAG systems?

## Hypotheses
H1: Hybrid lexical + dense retrieval improves retrieval of legally relevant evidence over BM25-only or dense-only retrieval.

H2: Structure-aware retrieval/summarization improves coverage of facts, issues, legal provisions, reasoning, and final decisions.

H3: Explicit evidence and citation verification reduces unsupported claims and fabricated legal references.

H4: Plain-language summaries can improve readability without materially reducing legal factuality when evidence grounding is enforced.

## Primary scope
Jurisdiction: India
Initial corpus: English Indian Supreme Court judgments
Initial study window: 2015–2025, subject to data quality and leakage checks
Main-study target: 1,000 usable judgments
Staged development: 10 → 50 → 200 → 1,000
Optional later expansion: 3,000+ and selected High Court data after MVP stability

## In scope
- judgment summarization
- legal question answering
- evidence retrieval
- citation verification
- claim/evidence verification
- abstention
- retrieval and generation evaluation
- human evaluation and ablations

## Out of scope
- autonomous legal advice
- replacement for lawyers
- confidential client documents
- unsupported claims of legal correctness

## System concept
Judgment
→ provenance-preserving ingestion
→ legal structure detection
→ evidence chunking
→ BM25 + dense retrieval
→ hybrid retrieval
→ optional reranking
→ evidence selection
→ LLM generation
→ citation verification
→ claim-level evidence verification
→ abstention when evidence is insufficient
→ cited answer / structured summary

## Core metadata
case_id
case_title
petitioner
respondent
court
decision_date
judge where available
citation
year
language
source_url
source_path
local_path
text_path
num_pages
num_words
available_languages
sha256
acquisition_method
acquisition_status
accessed_at
license_note

## Chunk record
chunk_id
case_id
page
paragraph
section
text
source_url/source reference

## QA record
question_id
case_id
question
gold_answer
evidence_chunk_ids
legal_references
question_type
difficulty
verified

## Evaluation
Retrieval:
Recall@k, MRR, nDCG

QA:
Exact Match where meaningful, token F1 where meaningful, semantic correctness

Citation:
validity, relevance, support, completeness, precision, recall

Factuality:
supported claim rate, unsupported claim rate

Summarization:
ROUGE-1/2/L, BERTScore, human correctness/completeness/clarity/legal coherence

Efficiency:
median latency, P95 latency, retrieval time, token usage, estimated cost

Robustness:
difficulty, domain, document length, temporal split, evidence-failure condition

## Baselines
A. Extractive summarization
B. Generic LLM without retrieval
C. Basic dense RAG
D. BM25-only RAG where appropriate
Proposed:
Hybrid BM25 + dense
+ legal structure
+ optional reranking
+ evidence/citation verification
+ claim verification
+ abstention

## Benchmark
Custom Indian legal QA/evidence benchmark:
- construction pool 300–500 judgments;
- stratified by year/domain where metadata permits;
- questions from Facts/Issues/Law/Reasoning/Decision;
- exact supporting paragraphs/chunks;
- human verification;
- difficulty labels;
- final test-set lock before tuning.

## Data integrity
Never assume metadata path is a PDF URL. Use verified index/archive/TAR/direct resolution. Log all failures.

## Safety
When supporting evidence is insufficient:
- abstain;
- say the answer cannot be verified;
- do not speculate.

## Research integrity
Never fabricate results, dataset counts, benchmark scores, annotations, literature statistics, or citations.
Never tune on the locked final test set.
Never claim novelty without literature evidence.

## Contribution framing
Potential contributions may include:
- evidence-grounded architecture for Indian judgments;
- empirical lexical/dense/hybrid retrieval comparison;
- legal-structure-aware evaluation;
- claim/citation verification evaluation;
- human-verified Indian QA/evidence benchmark;
- robustness and abstention analysis.

Only retain contribution claims supported by actual experiments.

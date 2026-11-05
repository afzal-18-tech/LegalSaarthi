"""Phase 15 summarization prompts. FROZEN after the first full val run; edits require a new version."""

SUMM_PROMPT_VERSION = "summ_v1"

SUMM_BLOCK_TEMPLATE = "[{block_id}] (section: {section}; pages: {pages})\n{text}"

GENERIC_SUMM_SYSTEM = (
    "You summarize Indian Supreme Court judgments. Use only the text provided. "
    "Do not invent facts, parties, statutes, or outcomes. Write about 150-200 words."
)
GENERIC_SUMM_USER = "Judgment text:\n{text}\n\nWrite a concise summary of this judgment."

STRUCTURED_SUMM_SYSTEM = (
    "You summarize Indian Supreme Court judgments using ONLY the numbered evidence blocks. "
    "Write exactly these four headings, each followed by 1-3 sentences:\n"
    "Facts:\nIssues:\nReasoning:\nDecision:\n"
    "End every sentence with the supporting block ID in square brackets, e.g. [p3_4]. "
    "Cite only IDs that appear in the evidence. If a heading has no support in the evidence, "
    "write 'Not stated in the provided text.' Do not invent facts, statutes, or outcomes. "
    "Total length about 150-220 words."
)
STRUCTURED_SUMM_USER = "Evidence blocks:\n{context}\n\nWrite the structured summary."

PLAIN_SUMM_SYSTEM = (
    "You explain Indian Supreme Court judgments to a non-lawyer using ONLY the numbered evidence blocks. "
    "Use short sentences and everyday words; explain any legal term you must use. "
    "Write exactly these four headings, each followed by 1-3 sentences:\n"
    "What happened:\nWhat the court had to decide:\nWhy the court decided as it did:\nFinal outcome:\n"
    "End every sentence with the supporting block ID in square brackets, e.g. [p3_4]. "
    "Cite only IDs that appear in the evidence. If a heading has no support, write "
    "'Not stated in the provided text.' Do not invent anything. This is not legal advice. "
    "Total length about 150-220 words."
)
PLAIN_SUMM_USER = "Evidence blocks:\n{context}\n\nWrite the plain-language summary."

STRUCTURED_HEADINGS = ["Facts", "Issues", "Reasoning", "Decision"]
PLAIN_HEADINGS = ["What happened", "What the court had to decide",
                  "Why the court decided as it did", "Final outcome"]

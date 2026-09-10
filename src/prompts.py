#!/usr/bin/env python3
"""
Shared prompt templates for the Bean Lab RAG system.

Design principles:
  1. Evidence-first chain of thought: extract → synthesize → conclude
  2. Explicit confidence labels: SUPPORTED / PARTIALLY_SUPPORTED / INFERRED / UNSUPPORTED
  3. Soft failure: always attempt a best-effort answer; label uncertainty explicitly
  4. Section-aware: chunks tagged with [METHODS] / [RESULTS] / [DISCUSSION] etc.
  5. Multi-part questions: decompose into sub-questions before answering
  6. Citation format: doi:10.XXXX/suffix (no generic [Source N] labels)
"""

import re
from typing import List, Dict, Optional, Tuple

DISTANCE_THRESHOLD = 0.85   # chunks above this are included but flagged as weak


# ── Section tag mapping ───────────────────────────────────────────────────────

SECTION_TAG_MAP = {
    "abstract":     "[ABSTRACT]",
    "introduction": "[INTRO]",
    "background":   "[INTRO]",
    "methods":      "[METHODS]",
    "materials":    "[METHODS]",
    "results":      "[RESULTS]",
    "discussion":   "[DISCUSSION]",
    "conclusion":   "[CONCLUSION]",
    "references":   "[REFERENCES]",
    "table":        "[TABLE]",
    "figure":       "[FIGURE]",
}


def get_section_tag(section: str) -> str:
    if not section:
        return ""
    s = section.lower()
    for key, tag in SECTION_TAG_MAP.items():
        if key in s:
            return tag
    return ""


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a structured context block.

    Each chunk is labeled with:
    - DOI and page
    - Section tag if available
    - Weak-evidence warning if distance > threshold
    """
    lines = []
    for i, c in enumerate(chunks, 1):
        # BM25-only chunks (hybrid retrieval, no dense match) have
        # distance=None — treat as worst-case rather than crashing the
        # comparison below (None > threshold raises TypeError).
        dist = c.get("distance")
        dist = 1.0 if dist is None else dist
        section_tag = get_section_tag(c.get("section", ""))
        quality = "⚠ WEAK EVIDENCE" if dist > DISTANCE_THRESHOLD else ""

        header_parts = [f"doi:{c['doi']}", f"p.{c['page']}"]
        if section_tag:
            header_parts.append(section_tag)
        if quality:
            header_parts.append(quality)

        lines.append(f"[{i}] {' | '.join(header_parts)}")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert scientific assistant specializing in bean and legume crop science. \
You answer questions in clear, natural prose — like a knowledgeable colleague explaining findings from the research literature.

WRITING STYLE:
• Write in flowing paragraphs as your default style
• Integrate citations naturally into sentences, e.g. "Studies show yields of 2–3 t/ha under rainfed conditions (doi:10.1234/example, p.4)"
• Use specific numbers, units, and percentages whenever the papers provide them
• If multiple papers agree, synthesize them into a single clear statement
• If papers disagree or evidence is limited, say so plainly in natural language
• For multi-part questions, use short subheadings only if it genuinely aids clarity
• Use a markdown table ONLY when the question asks for a comparison or summary AND the sources contain enough actual data to fill the table meaningfully

TABLE RULES — follow these strictly:
• Only include a row in a table if the source explicitly reports a value for that row — NEVER leave cells empty or write "not reported"
• Do not invent rows for categories that are not mentioned in the sources
• Every data cell must have a real number from a source; if no number exists, omit that row entirely
• Keep tables small and accurate — 3 accurate rows are better than 8 rows with empty cells
• Always follow a table with a brief paragraph summarising the key takeaway

CITATION RULES:
• Cite ONLY using DOI format: (doi:10.XXXX/suffix, p.N)
• NEVER use numbered references like [1], [2], [Source 1], [Document 2] — these are forbidden
• Only cite a source when it directly supports the specific claim in that sentence
• Sources marked ⚠ WEAK EVIDENCE should only be cited when no stronger source exists

TRIAL DATA (when a TRIAL DATA section is present above the retrieved sources):
• Trial data rows are exact numbers pulled from the corpus for the specific cultivar/location/year asked about — prefer them over paraphrasing a similar number from a retrieved passage
• These rows were extracted automatically from paper tables and may occasionally contain transcription errors — cite them the same way as any other source (doi:..., p.N) so the user can check the original if precision matters
• Use the surrounding literature sources to explain what the trial numbers mean, how they compare to other findings, and why they matter — the trial data gives precision, the literature gives interpretation

IMPORTANT:
• Never refuse to answer — give your best answer based on what the sources contain
• If the sources only partially address the question, answer what you can and briefly note what is not covered
• Do not mention confidence labels, retrieval systems, or internal scoring in your answer"""


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_messages(
    question: str,
    chunks: List[Dict],
    confidence: str = "",
    trial_data_block: str = "",
) -> List[Dict]:
    """
    Build OpenAI-style messages list for chat completion API (HF Spaces / any chat LLM).

    Args:
        question:         User's question
        chunks:           Retrieved and reranked chunks
        confidence:       Retrieval confidence label from BeanRetriever
        trial_data_block: Pre-formatted structured trial-data section — pass
                          structured_data.format_trial_results(rows) here, or
                          "" if the query router (Change 7) didn't activate
                          structured data for this question. Deliberately a
                          pre-formatted string rather than raw rows: the row
                          schema and its formatting belong to
                          structured_data.py, not here — this module only
                          places the block, it doesn't know the row shape.
    """
    context = build_context(chunks)

    confidence_note = ""
    if confidence in ("INFERRED", "UNSUPPORTED"):
        confidence_note = (
            "\n\nNOTE: The retrieved sources may only partially address this question. "
            "Answer as fully as you can from the evidence, and briefly note any gaps."
        )

    trial_section = f"{trial_data_block}\n\n" if trial_data_block else ""

    user_content = (
        f"{trial_section}"
        f"RETRIEVED SOURCES:\n\n{context}\n"
        f"{confidence_note}\n"
        f"QUESTION: {question}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_messages_with_history(
    question: str,
    chunks: List[Dict],
    confidence: str,
    history: List[Tuple[str, str]],
    trial_data_block: str = "",
) -> List[Dict]:
    """
    Like build_messages(), but prepends prior conversation turns (Change 8)
    so follow-up questions ("which of those are in navy bean?") have their
    antecedent available to the LLM.

    Args:
        history: (question, answer) tuples, oldest first, already trimmed by
                  the caller to whatever window it wants replayed (the Space
                  keeps the last 3). Only the plain question/answer text is
                  replayed — NOT that turn's retrieved sources or trial-data
                  block — so the prompt doesn't grow by a full context dump
                  per turn. This means the model reasons about earlier
                  answers by their conclusions, not by re-deriving them from
                  the original evidence each time; retrieval for the CURRENT
                  turn is what supplies fresh, current evidence.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for prev_q, prev_a in history:
        messages.append({"role": "user", "content": prev_q})
        messages.append({"role": "assistant", "content": prev_a})
    current_turn = build_messages(question, chunks, confidence, trial_data_block=trial_data_block)
    messages.append(current_turn[-1])
    return messages


def build_ollama_prompt(
    question: str,
    chunks: List[Dict],
    confidence: str = "",
    trial_data_block: str = "",
) -> str:
    """
    Build a single prompt string for Ollama (non-chat models or llama3 instruct format).

    Args: see build_messages() — same trial_data_block convention.
    """
    context = build_context(chunks)

    confidence_note = ""
    if confidence in ("INFERRED", "UNSUPPORTED"):
        confidence_note = (
            "\nNOTE: The retrieved sources may only partially address this question. "
            "Answer as fully as you can from the evidence, and briefly note any gaps.\n"
        )

    trial_section = f"{trial_data_block}\n\n" if trial_data_block else ""

    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{trial_section}"
        f"RETRIEVED SOURCES:\n\n{context}\n"
        f"{confidence_note}"
        f"QUESTION: {question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )


def format_references(chunks: List[Dict], titles: Optional[Dict[str, str]] = None) -> str:
    """
    Build a clean deduplicated reference list from retrieved chunks.
    Only includes chunks below the distance threshold.

    titles: optional {source_file: title} lookup — data/paper_titles.json,
    LLM-extracted by scripts/extract_paper_titles.py (see that script's
    docstring for why a real title needs an LLM, not a heuristic; this is
    the same file that already powers the GraphRAG paper-graph's node
    labels). When a chunk's source file has a known title, shows it with
    the DOI as a trailing link: "Paper Title (doi:X)". Falls back to just
    the DOI as the link text — the original format — for any chunk whose
    paper isn't in the titles lookup (e.g. papers outside the 964-paper
    GraphRAG corpus, or the ~2 that failed extraction).
    """
    titles = titles or {}
    seen = set()
    refs = []
    for c in chunks:
        doi = c.get("doi", "")
        source = c.get("source", "")
        dist = c.get("distance")
        dist = 1.0 if dist is None else dist
        if doi and doi not in seen:
            seen.add(doi)
            flag = " *(weak evidence)*" if dist > DISTANCE_THRESHOLD else ""
            title = titles.get(source)
            link = f"[doi:{doi}](https://doi.org/{doi})"
            refs.append(f"• {title} ({link}){flag}" if title else f"• {link}{flag}")
    if not refs:
        return ""
    return "**References:**\n" + "\n".join(refs)


# Inline citations follow CITATION RULES above: (doi:10.XXXX/suffix, p.N),
# deliberately NOT [1]/[2]/[Source N] — see commit 41393b5, which forbade
# numbered references specifically because an LLM can mislabel or invent a
# bracket number with no real chunk behind it, while a DOI it writes is
# always one it actually saw in the retrieved context. This function does
# NOT touch that — it's a pure display transform, applied to the model's
# output after generation, turning the raw "(doi:X, p.N)" text a user can't
# click into an actual link they can, without changing what gets cited or
# how the LLM decides to cite it.
_INLINE_CITATION_RE = re.compile(r"\(doi:([^,\)]+),\s*p\.(\d+)\)")


def linkify_citations(answer: str) -> str:
    """
    (doi:10.2135/cropsci2004.1901, p.14) -> ([p.14](https://doi.org/10.2135/cropsci2004.1901))
    Leaves the text unchanged if it doesn't match this exact pattern (e.g. a
    citation format the LLM didn't quite follow) rather than risk mangling it.
    """
    return _INLINE_CITATION_RE.sub(r"([p.\2](https://doi.org/\1))", answer)
